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

def get_clean_question_numbers(image_paths):
    """
    Asks Groq to act as a STRICT OCR FILTER.
    Only returns numbers that belong to QUESTIONS.
    Explicitly tells Groq to read text and exclude Solutions.
    """
    if not client or not image_paths: return None
    
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    try:
        message_content = [
            {
                "type": "text",
                "text": """
                You are an OCR Filter. Read these exam pages.
                
                YOUR TASK:
                Return a JSON list of Question Numbers (Integers) for ACTUAL QUESTIONS only.
                
                STRICT RULES:
                1. If a number is followed by "Solution", "Ans", "Correct Answer", "Hint", or "Explanation" -> IGNORE IT.
                2. If a number is just a Page Number or Marks -> IGNORE IT.
                3. ONLY return the number if it starts a Question Statement.
                
                FORMAT:
                {
                    "img_0": [1, 2, 3],  // Found Q1, Q2, Q3 (Ignored Q1 Solution)
                    "img_1": [4, 5]
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
            temperature=0.1, max_tokens=1000, top_p=1, stream=False,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        return data

    except Exception as e:
        log(f"❌ Groq OCR Failed: {str(e)}")
        return None

# --- HELPER FUNCTIONS ---
def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_questions_groq_guided(doc, page_map_guidance):
    extracted_data = []
    
    # Simple Start-of-block Regex
    # Matches "1.", "Q1", "Q.1", "1)" at the start of the text block
    regex_pat = r"^\s*(?:Q|Question|Que|No)?[\.\s\-]?\s*(\d+)[\.\)\-\:]?"
    
    top_m = 0 
    bot_m = 40 # Footer buffer
    
    for page_num in range(len(doc)):
        # What did Groq say about this page?
        allowed_numbers = page_map_guidance.get(page_num, [])
        if not allowed_numbers: continue
            
        page = doc[page_num]
        height = page.rect.height
        blocks = page.get_text("blocks")
        
        # Avoid duplicate captures on same page
        found_on_page = set()

        for b in blocks:
            text = b[4].strip()
            bbox = b[:4]
            x0, y0, x1, y1 = bbox
            
            if y1 > height - bot_m: continue
            
            # --- BLIND OBEDIENCE CHECK ---
            # 1. Check if block starts with a number
            match = re.match(regex_pat, text, re.IGNORECASE)
            if match:
                try:
                    val = int(match.group(1))
                    
                    # 2. ASK GROQ: "Is this number in your list?"
                    if val in allowed_numbers:
                        # 3. YES -> Crop it.
                        if val not in found_on_page:
                            extracted_data.append({
                                "label": str(val), 
                                "x0": x0, "y0": y0, 
                                "page": page_num, 
                                "bbox": bbox
                            })
                            found_on_page.add(val)
                    # 4. NO -> Ignore it (It must be a solution or page number)
                except: continue

    return extracted_data

def process_cbt_logic(pdf_path):
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    FULL_PAGE_GUIDANCE = {}
    BATCH_SIZE = 2 # Process 2 pages at a time
    
    # --- STEP 1: ASK GROQ (THE GOD) ---
    for i in range(0, total_pages, BATCH_SIZE):
        batch_indices = range(i, min(i + BATCH_SIZE, total_pages))
        img_paths = []
        
        log(f"🤖 Groq OCR Processing Pages {[b+1 for b in batch_indices]}...")
        
        for p_idx in batch_indices:
            pix = doc[p_idx].get_pixmap(dpi=100)
            path = os.path.join(UPLOAD_FOLDER, f"batch_page_{p_idx}.jpg")
            pix.save(path)
            img_paths.append(path)
            
        # Get Clean List (No Solutions)
        batch_data = get_clean_question_numbers(img_paths)
        
        if batch_data:
            for rel_idx, key in enumerate(sorted(batch_data.keys())):
                if rel_idx < len(batch_indices):
                    global_page_idx = batch_indices[rel_idx]
                    # Parse Groq's list safely
                    raw_list = batch_data[key]
                    clean_qs = []
                    for x in raw_list:
                        if isinstance(x, int): clean_qs.append(x)
                        elif isinstance(x, str) and x.isdigit(): clean_qs.append(int(x))
                    
                    if clean_qs:
                        FULL_PAGE_GUIDANCE[global_page_idx] = clean_qs
                        log(f"   -> Page {global_page_idx+1}: Authorized Qs {clean_qs}")
        
        time.sleep(0.5)

    # --- STEP 2: EXECUTE ORDERS ---
    final_questions = extract_questions_groq_guided(doc, FULL_PAGE_GUIDANCE)

    # --- FAILSAFE (Only if Groq fails completely) ---
    if not final_questions:
        log("⚠️ Groq returned nothing. Trying Brute Force (All Numbers).")
        # Map every page to all possible numbers 1-200 to catch *something*
        fallback_map = {p: list(range(1, 200)) for p in range(len(doc))}
        final_questions = extract_questions_groq_guided(doc, fallback_map)

    log(f"✅ Final Questions Extracted: {len(final_questions)}")

    # --- JSON & CROP ---
    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": "Team_Stark_V30_GroqGod"
    }

    final_questions.sort(key=lambda x: (x["page"], x["y0"]))

    for i, q in enumerate(final_questions):
        pg_h = doc[q["page"]].rect.height
        pg_w = doc[q["page"]].rect.width
        
        # Crop full width
        x1 = 0
        x2 = pg_w
        json_x1 = 5
        json_x2 = 995

        next_q_y = pg_h - 40 
        
        for j in range(i + 1, len(final_questions)):
            if final_questions[j]["page"] == q["page"]:
                next_q_y = final_questions[j]["y0"] - 10
                break

        y1_crop = max(0, q["y0"] - 10) # Start slightly above number
        y2_crop = min(pg_h, next_q_y + 5)
        
        # Minimum height safety
        if y2_crop - y1_crop < 50: y2_crop = y1_crop + 150

        data_json["pdfCropperData"][SUBJECT_NAME][SECTION_NAME][str(q["label"])] = {
            "que": q["label"], "type": "mcq", "marks": {"cm": 4, "im": -1},
            "pdfData": [{
                "x1": json_x1, "x2": json_x2, 
                "y1": round(((y1_crop)/pg_h)*1000), 
                "y2": round(((y2_crop)/pg_h)*1000), 
                "page": q["page"] + 1
            }],
            "answerOptions": "4"
        }

        page = doc[q["page"]]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        scale_w = pix.width / pg_w
        scale_h = pix.height / pg_h
        
        crop_box = (0, y1_crop * scale_h, pg_w * scale_w, y2_crop * scale_h)
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
    return "Team Stark V30 (Groq is God) 🚀"

@app.route('/process', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    if not is_authorized(request):
        return jsonify({"error": "Unauthorized"}), 403

    log("🔵 New Request")
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
