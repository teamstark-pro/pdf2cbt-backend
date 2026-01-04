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
import random
import google.generativeai as genai

app = Flask(__name__)

# --- 🔥 SECURITY & CORS 🔥 ---
# Allow Frontend to send custom header 'x-stark-secret'
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
    # If secret is not set in env, allow everyone (Debug mode)
    if STARK_SECRET == "open_access_mode": return True
    
    # Check Header
    client_key = req.headers.get("x-stark-secret")
    if client_key == STARK_SECRET:
        return True
    return False

# --- 🔑 KEY ROTATION ---
RAW_KEYS = os.environ.get("GEMINI_API_KEYS", "")
API_KEY_POOL = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

def configure_random_key():
    if not API_KEY_POOL:
        # Fallback to single key if pool is empty
        single_key = os.environ.get("GEMINI_API_KEY")
        if single_key:
            genai.configure(api_key=single_key)
            return True
        return False
    
    selected_key = random.choice(API_KEY_POOL)
    genai.configure(api_key=selected_key)
    log(f"🔑 Rotated Key ending in ...{selected_key[-4:]}")
    return True

# --- 🧠 SUPER AI ANALYSIS ---
def get_ai_config(image_paths):
    if not configure_random_key(): return None
    
    models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-flash-latest']
    
    for model_name in models_to_try:
        try:
            log(f"🤖 Analyzing with {model_name}...")
            model = genai.GenerativeModel(model_name)
            content_parts = []
            for path in image_paths:
                content_parts.append(genai.upload_file(path))
            
            # --- THE GOD MODE PROMPT ---
            prompt = """
            Analyze these exam pages deeply. I need to crop individual questions.
            Return ONLY a JSON object with these fields:
            
            1. "top_margin": Header height in pixels to safely ignore (e.g., 60).
            2. "bottom_margin": Footer height in pixels (e.g., 50).
            3. "regex_pattern": Python Regex to match the Question Number at the start of a line.
               - Examples: "^Q\\.?[\\s-]?\\s?(\\d+)[\\.\\)]" or "^(\\d+)\\s*[\\.]"
            4. "is_two_column": Boolean (true/false). Does the layout look like 2 columns?
            5. "min_question_number": Integer. The first question number visible (usually 1).
            
            Be precise. If the page is split vertically into two distinct columns of questions, set is_two_column to true.
            """
            content_parts.append(prompt)
            
            result = model.generate_content(content_parts)
            text_resp = result.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text_resp)
            log(f"✅ AI Insight: {data}")
            return data
        except Exception as e:
            log(f"❌ {model_name} Error: {str(e)[:50]}...")
            continue
            
    return None

# --- HELPER FUNCTIONS ---
def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_questions_with_strategy(doc, strategy_name, config):
    extracted_data = []
    
    # Unpack Config
    top_m = int(config.get("top_margin", 50))
    bot_m = int(config.get("bottom_margin", 50))
    regex = config.get("regex_pattern")
    force_two_col = config.get("is_two_column", False)
    
    log(f"🔄 Executing {strategy_name} | Regex: {regex} | 2-Col: {force_two_col}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        height = page.rect.height
        mid_x = page.rect.width / 2
        blocks = page.get_text("blocks")
        
        for b in blocks:
            text = b[4].strip()
            bbox = b[:4]
            x0 = bbox[0]

            if bbox[1] < top_m or bbox[3] > height - bot_m: continue 
            if text.startswith("["): continue 
            if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.I): continue

            # If AI says it's definitely NOT 2 column, enforce strict full width check
            # Otherwise, keep loose check
            if not force_two_col and "PANIC" not in strategy_name:
                 # If text is suspiciously in the middle, might be garbage
                 pass 

            try:
                q_match = re.search(regex, text, re.IGNORECASE)
                if q_match:
                    if any(x in text for x in ["Answer", "Solution", "Page", "Total"]): continue
                    q_no_str = next((g for g in q_match.groups() if g), None)
                    if not q_no_str: continue
                    
                    try:
                        q_val = int(q_no_str)
                        if q_val <= 0 or q_val > 500: continue
                    except: continue

                    extracted_data.append({
                        "label": q_no_str, 
                        "x0": bbox[0], "y0": bbox[1], 
                        "page": page_num, 
                        "bbox": bbox
                    })
            except: continue
            
    return extracted_data

def process_cbt_logic(pdf_path):
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    doc = fitz.open(pdf_path)
    
    # Generate Images (Low DPI)
    pages_to_check = min(2, len(doc))
    img_paths = []
    for i in range(pages_to_check):
        pix = doc[i].get_pixmap(dpi=100)
        p_path = os.path.join(UPLOAD_FOLDER, f"analyze_page_{i}.jpg")
        pix.save(p_path)
        img_paths.append(p_path)

    # --- STRATEGY BUILDER ---
    strategies = []
    
    # 1. AI STRATEGY (High Precision)
    ai_data = get_ai_config(img_paths)
    if ai_data:
        strategies.append({
            "name": "GEMINI_GOD_MODE",
            "top_margin": ai_data.get("top_margin"),
            "bottom_margin": ai_data.get("bottom_margin"),
            "regex_pattern": ai_data.get("regex_pattern"),
            "is_two_column": ai_data.get("is_two_column")
        })

    # 2. FALLBACK STRATEGIES
    strategies.append({
        "name": "FALLBACK_STD",
        "top_margin": 50, "bottom_margin": 50,
        "regex_pattern": r"^(?:Q|Question|Que|No|Problem)?[\.\s\-]?\s*(\d+)[\.\)\-]",
        "is_two_column": False # Assume single unless proven otherwise
    })
    
    strategies.append({
        "name": "PANIC_MODE",
        "top_margin": 0, "bottom_margin": 0,
        "regex_pattern": r"^(\d+)",
        "is_two_column": False
    })

    final_questions = []
    used_strategy = "NONE"
    active_config = {}

    for strat in strategies:
        questions = extract_questions_with_strategy(doc, strat["name"], strat)
        if len(questions) >= 2:
            log(f"✅ LOCKED! Found {len(questions)} qs using {strat['name']}")
            final_questions = questions
            used_strategy = strat["name"]
            active_config = strat
            break 

    if not final_questions:
        raise Exception("Failed to crop ANY questions.")

    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": f"Team_Stark_{used_strategy}"
    }

    # --- CROP LOGIC WITH AI INTELLIGENCE ---
    force_two_col = active_config.get("is_two_column", False)

    for i, q in enumerate(final_questions):
        pg_h = doc[q["page"]].rect.height
        pg_w = doc[q["page"]].rect.width
        mid_x = pg_w / 2
        
        # 1. Determine Column Status
        # If AI said "It's Two Column", we respect that.
        # OR if the question is literally on the right side.
        is_right_col = q["x0"] > mid_x
        
        # Determine crop width
        if force_two_col or is_right_col:
            # Strictly split page in half
            x1 = 0 if not is_right_col else mid_x
            x2 = mid_x if not is_right_col else pg_w
            json_x1 = 5 if not is_right_col else 505
            json_x2 = 495 if not is_right_col else 995
        else:
            # Single Column (Full Width)
            x1 = 0
            x2 = pg_w
            json_x1 = 5
            json_x2 = 995

        next_q_y = pg_h - 50 
        
        if i + 1 < len(final_questions):
            nq = final_questions[i+1]
            if nq["page"] == q["page"]:
                 # If in same column zone, crop till next question
                 if (force_two_col and (is_right_col == (nq["x0"] > mid_x))) or not force_two_col:
                     next_q_y = nq["y0"] - 15

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
    return "Stark Secured Backend V13 (Auth + God Mode) 🔒"

@app.route('/process', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    # 🔒 AUTH CHECK
    if not is_authorized(request):
        log("⛔ Unauthorized Access Attempt Blocked")
        return jsonify({"error": "Unauthorized: Invalid Stark Secret"}), 403

    log("🔵 New Authorized Request Received")
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
