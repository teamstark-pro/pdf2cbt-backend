from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import os
import json
import hashlib
import shutil
import re
from PIL import Image
import zipfile
import io
import sys
import gc
import base64
import time
from groq import Groq

app = Flask(__name__)

# --- 🔥 CORS & SECURITY 🔥 ---
CORS(app, resources={r"/*": {"origins": "*"}}, 
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["*"],
     expose_headers=["Content-Disposition", "Content-Type"])

EXPORT_DIR = "/tmp/cbt_master_package"
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 🔍 LOGGING HELPER ---
def log(msg):
    print(f"[STARK LOG] {msg}", file=sys.stdout, flush=True)

# --- 🔐 SECURITY CHECK ---
STARK_SECRET = os.environ.get("STARK_SECRET_KEY", "open_access_mode")

def is_authorized(req):
    if STARK_SECRET == "open_access_mode": return True
    client_key = req.headers.get("x-stark-secret")
    return client_key == STARK_SECRET

# --- ⚡ GROQ CLIENT ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_questions_per_page_batch(image_paths):
    """
    Sends a batch of images (max 2) to Groq.
    Asks ONLY for the Question Numbers present, explicitly ignoring solutions.
    """
    if not client or not image_paths: return None
    
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    try:
        message_content = [
            {
                "type": "text",
                "text": """
                Analyze these exam pages. Identify ONLY the Question Numbers for actual QUESTIONS.
                
                CRITICAL RULES:
                1. IGNORE any block labeled "Solution", "Answer", "Correct Answer", "Sol.", "Ans".
                   (Example: If you see "Q18. Solution", DO NOT include 18).
                2. IGNORE page numbers, marks, or section headers.
                3. ONLY return integers of valid questions to be solved.
                
                Return JSON mapping image index to list of numbers:
                {
                    "img_0": [1, 2, 3],
                    "img_1": [] 
                }
                
                Output JSON ONLY.
                """
            }
        ]
        
        for path in image_paths:
            base64_img = encode_image(path)
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": message_content}],
            temperature=0.1, max_tokens=500, top_p=1, stream=False,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        return data

    except Exception as e:
        log(f"❌ Groq Batch Failed: {str(e)}")
        return None

# --- HELPER FUNCTIONS ---
def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_questions_with_ai_guide(doc, page_map_guidance):
    extracted_data = []
    
    # 1. Standard Regex for hunting
    regex_list = [
        r"^(?:Q|Question|Que|No|Problem)?[\.\s\-]?\s*(\d+)[\.\)\-]",
        r"^(\d+)\s*[\.\)]"
    ]
    
    # 2. THE ANTI-SOLUTION SHIELD (Python Layer)
    # If text contains these, we skip it even if AI said "Yes"
    BAD_KEYWORDS = [
        "Solution", "Sol.", "Answer", "Ans.", "Correct Answer", 
        "Explanation", "Hint", "Step 1", "Hence", "Therefore"
    ]
    
    top_m = 50
    bot_m = 50
    
    for page_num in range(len(doc)):
        allowed_numbers = page_map_guidance.get(page_num, [])
        if not allowed_numbers:
            continue
            
        page = doc[page_num]
        height = page.rect.height
        blocks = page.get_text("blocks")
        
        for b in blocks:
            text = b[4].strip()
            bbox = b[:4]
            x0, y0, x1, y1 = bbox
            
            # Margins
            if y0 < top_m or y1 > height - bot_m: continue
            
            # --- 🔥 FILTER: REJECT SOLUTIONS 🔥 ---
            if any(bad in text for bad in BAD_KEYWORDS):
                # log(f"   -> Skipped Solution Block: '{text[:20]}...'")
                continue

            # --- HYBRID CHECK ---
            match_found = False
            for pat in regex_list:
                try:
                    q_match = re.search(pat, text, re.IGNORECASE)
                    if q_match:
                        q_no_str = next((g for g in q_match.groups() if g), None)
                        if not q_no_str: continue
                        
                        q_val = int(q_no_str)
                        
                        # Verify against AI List
                        if q_val in allowed_numbers:
                            extracted_data.append({
                                "label": q_no_str, 
                                "x0": x0, "y0": y0, 
                                "page": page_num, 
                                "bbox": bbox
                            })
                            match_found = True
                            break
                except: continue
                if match_found: break

    return extracted_data

def process_cbt_logic(pdf_path):
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # --- STEP 1: BATCH PROCESSING WITH GROQ ---
    FULL_PAGE_GUIDANCE = {}
    BATCH_SIZE = 2
    
    for i in range(0, total_pages, BATCH_SIZE):
        batch_indices = range(i, min(i + BATCH_SIZE, total_pages))
        img_paths = []
        
        log(f"🤖 Processing Pages {[b+1 for b in batch_indices]}...")
        
        for p_idx in batch_indices:
            pix = doc[p_idx].get_pixmap(dpi=100)
            path = os.path.join(UPLOAD_FOLDER, f"batch_page_{p_idx}.jpg")
            pix.save(path)
            img_paths.append(path)
            
        batch_data = get_questions_per_page_batch(img_paths)
        
        if batch_data:
            for rel_idx, key in enumerate(sorted(batch_data.keys())):
                if rel_idx < len(batch_indices):
                    global_page_idx = batch_indices[rel_idx]
                    found_qs = [int(x) for x in batch_data[key] if isinstance(x, (int, str)) and str(x).isdigit()]
                    FULL_PAGE_GUIDANCE[global_page_idx] = found_qs
                    if found_qs:
                        log(f"   -> Page {global_page_idx+1}: Qs {found_qs}")
        
        time.sleep(0.5)

    # --- STEP 2: EXTRACT USING AI GUIDANCE + FILTER ---
    final_questions = extract_questions_with_ai_guide(doc, FULL_PAGE_GUIDANCE)

    # --- FAILSAFE ---
    if not final_questions:
        log("⚠️ AI found nothing. Checking for text-only PDF or Encryption.")
        raise Exception("No valid questions found (Solutions skipped).")

    log(f"✅ Total Verified Questions: {len(final_questions)}")

    # --- JSON & CROP ---
    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": "Team_Stark_V26_AntiSolution"
    }

    final_questions.sort(key=lambda x: (x["page"], x["y0"]))

    for i, q in enumerate(final_questions):
        pg_h = doc[q["page"]].rect.height
        pg_w = doc[q["page"]].rect.width
        
        x1 = 0
        x2 = pg_w
        json_x1 = 5
        json_x2 = 995

        next_q_y = pg_h - 50 
        
        for j in range(i + 1, len(final_questions)):
            if final_questions[j]["page"] == q["page"]:
                next_q_y = final_questions[j]["y0"] - 15
                break

        y1_crop = max(0, q["y0"] - 30)
        y2_crop = min(pg_h, next_q_y + 10)
        if y2_crop <= y1_crop: y2_crop = y1_crop + 200

        data_json["pdfCropperData"][SUBJECT_NAME][SECTION_NAME][str(q["label"])] = {
            "que": q["label"], "type": "mcq", "marks": {"cm": 4, "im": -1},
            "pdfData": [{
                "x1": json_x1, "x2": json_x2, 
                "y1": round(((y1_crop + 5)/pg_h)*1000), 
                "y2": round(((y2_crop - 5)/pg_h)*1000), 
                "page": q["page"] + 1
            }],
            "answerOptions": "4"
        }

        page = doc[q["page"]]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        scale_w = pix.width / pg_w
        scale_h = pix.height / pg_h
        
        crop_box = (x1 * scale_w, y1_crop * scale_h, x2 * scale_w, y2_crop * scale_h)
        try:
            cropped = img.crop(crop_box)
            img_name = f"{SECTION_NAME}__--__{q['label']}__--__1.png"
            cropped.save(os.path.join(EXPORT_DIR, img_name))
        except: pass
        del img, pix

    doc.close()
    gc.collect()

    with open(os.path.join(EXPORT_DIR, "data.json"), "w") as f:
        json.dump(data_json, f, indent=2)

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(EXPORT_DIR):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    
    memory_file.seek(0)
    return memory_file

# --- FLASK ROUTES ---
@app.route('/')
def home():
    return "Team Stark V26 (Anti-Solution Shield) 🚀"

@app.route('/process', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    if not is_authorized(request):
        return jsonify({"error": "Unauthorized"}), 403

    log("🔵 New Request (Page-by-Page + Shield)")
    if 'pdf' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['pdf']
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        try:
            zip_buffer = process_cbt_logic(filepath)
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='Stark_Result.zip')
        except Exception as e:
            log(f"🔥 FATAL: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
