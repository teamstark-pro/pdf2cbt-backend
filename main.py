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
    if not client or not image_paths: return None
    
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    try:
        message_content = [
            {
                "type": "text",
                "text": """
                Analyze these exam pages. List ONLY the Question Numbers found.
                
                RULES:
                1. Look for numbers at the START of blocks (e.g. "1.", "Q1", "5)").
                2. IGNORE numbers inside Solution/Answer/Hint blocks.
                3. Return JSON: { "img_0": [1, 2, 3], "img_1": [4, 5] }
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

def extract_questions_strict_ai(doc, page_map_guidance):
    extracted_data = []
    
    # 🔥 STRICT REGEX (Must be at Start of String)
    # ^ ensures we don't match "20" inside "Length is 20cm"
    regex_list = [
        r"^\s*(?:Q|Question|Que|No)[\.\s\-]?\s*(\d+)",  # Matches "Q. 20", "Question 20"
        r"^\s*(\d+)[\.\)\-\:]",                         # Matches "20.", "20)", "20-"
        r"^\s*(\d+)\s*$"                                # Matches standalone "20"
    ]
    
    # ❌ ANTI-SOLUTION SHIELD
    BAD_KEYWORDS = ["Solution", "Detailed Solution", "Correct Answer", "Explanation", "Hint:", "Ans."]
    
    top_m = 0 
    bot_m = 40 # Footer
    
    for page_num in range(len(doc)):
        allowed_numbers = page_map_guidance.get(page_num, [])
        if not allowed_numbers: continue
            
        page = doc[page_num]
        height = page.rect.height
        blocks = page.get_text("blocks")
        
        # Track what we found to avoid duplicates
        found_on_page = set()

        for b in blocks:
            text = b[4].strip()
            bbox = b[:4]
            x0, y0, x1, y1 = bbox
            
            if y1 > height - bot_m: continue
            if y0 < top_m: continue
            
            # 1. Skip if Solution
            if any(bad in text for bad in BAD_KEYWORDS): continue

            # 2. Strict Start-of-Block Match
            matched_num = None
            for pat in regex_list:
                # Using re.match or ^ anchor in search
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        # Only accept if Groq said it's here
                        if val in allowed_numbers:
                            matched_num = str(val)
                            # Only add if we haven't processed this Q yet (First occurrence wins)
                            if val not in found_on_page:
                                found_on_page.add(val)
                                break
                            else:
                                matched_num = None # Duplicate on page (maybe header repetition)
                    except: continue
            
            if matched_num:
                extracted_data.append({
                    "label": matched_num, 
                    "x0": x0, "y0": y0, 
                    "page": page_num, 
                    "bbox": bbox
                })

    return extracted_data

def process_cbt_logic(pdf_path):
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    FULL_PAGE_GUIDANCE = {}
    BATCH_SIZE = 2
    
    # --- STEP 1: GROQ SCANNING ---
    for i in range(0, total_pages, BATCH_SIZE):
        batch_indices = range(i, min(i + BATCH_SIZE, total_pages))
        img_paths = []
        
        log(f"🤖 Scanning Pages {[b+1 for b in batch_indices]}...")
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
                    if found_qs:
                        FULL_PAGE_GUIDANCE[global_page_idx] = found_qs
                        log(f"   -> Page {global_page_idx+1}: {found_qs}")
        time.sleep(0.5)

    # --- STEP 2: EXTRACT ---
    final_questions = extract_questions_strict_ai(doc, FULL_PAGE_GUIDANCE)

    # --- FAILSAFE ---
    if not final_questions:
        log("⚠️ No Strict Matches. Falling back to Loose Search (Desperation Mode).")
        # Reuse magnet logic (v28 style) only if strict failed
        # But limited to AI numbers
        final_questions = extract_questions_strict_ai(doc, FULL_PAGE_GUIDANCE) # (Strict failed, so empty)
        # Try brute force all numbers 1-100 if completely empty
        if not final_questions:
             log("💀 AI Guidance failed. Brute forcing...")
             # Just map every page to 1-100 to catch anything
             final_questions = extract_questions_strict_ai(doc, {p: list(range(1, 100)) for p in range(len(doc))})


    log(f"✅ Extracted {len(final_questions)} Questions")

    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": "Team_Stark_V29_Strict"
    }

    final_questions.sort(key=lambda x: (x["page"], x["y0"]))

    for i, q in enumerate(final_questions):
        pg_h = doc[q["page"]].rect.height
        pg_w = doc[q["page"]].rect.width
        
        next_q_y = pg_h - 40 
        
        for j in range(i + 1, len(final_questions)):
            if final_questions[j]["page"] == q["page"]:
                next_q_y = final_questions[j]["y0"] - 10
                break

        y1_crop = max(0, q["y0"] - 10) 
        y2_crop = min(pg_h, next_q_y + 5)
        
        if y2_crop - y1_crop < 50: y2_crop = y1_crop + 150

        data_json["pdfCropperData"][SUBJECT_NAME][SECTION_NAME][str(q["label"])] = {
            "que": q["label"], "type": "mcq", "marks": {"cm": 4, "im": -1},
            "pdfData": [{
                "x1": 5, "x2": 995, 
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
    return "Team Stark V29 (Strict Start-Block Logic) 🚀"

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
