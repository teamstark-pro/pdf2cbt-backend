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
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

EXPORT_DIR = "/tmp/cbt_master_package"
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 🔍 LOGGING HELPER ---
def log(msg):
    print(f"[STARK LOG] {msg}", file=sys.stdout, flush=True)

# --- 🔑 CONFIG & DIAGNOSTICS ---
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

# --- 🧠 AI ANALYSIS ---
def get_ai_config(image_paths):
    if not GENAI_API_KEY: return None
    
    # 🔥 UPDATED MODELS BASED ON YOUR LOGS 🔥
    models_to_try = [
        'gemini-2.0-flash',       # Latest Fast Model
        'gemini-2.5-flash',       # Preview Model
        'gemini-flash-latest',    # Auto-latest
        'gemini-2.0-flash-lite'   # Super cheap/fast
    ]
    
    for model_name in models_to_try:
        try:
            log(f"🤖 Attempting AI Model: {model_name}...")
            model = genai.GenerativeModel(model_name)
            content_parts = []
            for path in image_paths:
                content_parts.append(genai.upload_file(path))
            
            prompt = """
            Analyze these exam pages. Return ONLY JSON.
            1. "top_margin": Header height px (e.g. 60).
            2. "bottom_margin": Footer height px (e.g. 50).
            3. "regex_pattern": Regex for Question Start. 
               Ex: "^Q\\.?[\\s-]?\\s?(\\d+)[\\.\\)]" or "^(\\d+)\\s*[\\.]"
            """
            content_parts.append(prompt)
            
            result = model.generate_content(content_parts)
            text_resp = result.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text_resp)
            log(f"✅ AI Success ({model_name}): {data}")
            return data
        except Exception as e:
            log(f"❌ {model_name} Error: {str(e)[:100]}") # Keep log short
            continue
            
    return None

# --- HELPER FUNCTIONS ---
def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_questions_with_strategy(doc, strategy_name, top_m, bot_m, regex_pattern):
    extracted_data = []
    log(f"🔄 Strategy: {strategy_name} | Regex: {regex_pattern} | Margins: {top_m}/{bot_m}")

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

            # Indentation Check (Loose for Panic Mode)
            if "PANIC" not in strategy_name:
                if x0 > 80 and x0 < mid_x: continue 
                if x0 > mid_x + 80: continue

            try:
                q_match = re.search(regex_pattern, text, re.IGNORECASE)
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
    
    # Images for AI
    pages_to_check = min(2, len(doc))
    img_paths = []
    for i in range(pages_to_check):
        pix = doc[i].get_pixmap(dpi=150)
        p_path = os.path.join(UPLOAD_FOLDER, f"analyze_page_{i}.jpg")
        pix.save(p_path)
        img_paths.append(p_path)

    # Strategies
    strategies = []
    
    # Strategy 1: AI (With corrected Model Names)
    ai_data = get_ai_config(img_paths)
    if ai_data:
        strategies.append({
            "name": "GEMINI_AI",
            "top": int(ai_data.get("top_margin", 60)),
            "bot": int(ai_data.get("bottom_margin", 50)),
            "regex": ai_data.get("regex_pattern")
        })

    # Strategy 2: Standard Fallback
    strategies.append({
        "name": "FALLBACK_STANDARD",
        "top": 50, "bot": 50,
        "regex": r"^(?:Q|Question|Que|No|Problem)?[\.\s\-]?\s*(\d+)[\.\)\-]"
    })
    
    # Strategy 3: Panic Mode
    strategies.append({
        "name": "PANIC_MODE",
        "top": 0, "bot": 0,
        "regex": r"^(\d+)"
    })

    final_questions = []
    used_strategy = "NONE"
    
    for strat in strategies:
        questions = extract_questions_with_strategy(doc, strat["name"], strat["top"], strat["bot"], strat["regex"])
        if len(questions) >= 2:
            log(f"✅ LOCKED! Found {len(questions)} questions using {strat['name']}")
            final_questions = questions
            used_strategy = strat["name"]
            break 

    if not final_questions:
        raise Exception("Failed to crop ANY questions.")

    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": f"Team_Stark_{used_strategy}"
    }

    # --- PROCESS AND CROP ---
    for i, q in enumerate(final_questions):
        pg_h = doc[q["page"]].rect.height
        pg_w = doc[q["page"]].rect.width
        mid_x = pg_w / 2
        
        # --- SMART COLUMN DETECTION ---
        # Check if ANY question on this page is on the right side
        page_qs = [xq for xq in final_questions if xq["page"] == q["page"]]
        has_right_col = any(xq["x0"] > mid_x + 20 for xq in page_qs)
        
        # If no questions on right side, it's SINGLE COLUMN -> Use Full Width
        is_right_col = q["x0"] > mid_x
        
        # Determine X Coordinates
        if has_right_col:
            # Multi-column Logic
            x1 = 0 if not is_right_col else mid_x
            x2 = mid_x if not is_right_col else pg_w
            json_x1 = 5 if not is_right_col else 505
            json_x2 = 495 if not is_right_col else 995
        else:
            # Single Column Logic (FULL WIDTH)
            x1 = 0
            x2 = pg_w
            json_x1 = 5
            json_x2 = 995

        # Determine Y Coordinates
        next_q_y = pg_h - 50 
        
        if i + 1 < len(final_questions):
            nq = final_questions[i+1]
            if nq["page"] == q["page"]:
                 if not has_right_col or (is_right_col == (nq["x0"] > mid_x)):
                     next_q_y = nq["y0"] - 15

        # Coordinate Guard
        y1_crop = max(0, q["y0"] - 30)
        y2_crop = min(pg_h, next_q_y + 10)
        if y2_crop <= y1_crop: y2_crop = y1_crop + 200

        # JSON Coords
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

        # Image Crop
        page = doc[q["page"]]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        scale_w = pix.width / pg_w
        scale_h = pix.height / pg_h
        
        crop_box = (
            x1 * scale_w, 
            y1_crop * scale_h,
            x2 * scale_w, 
            y2_crop * scale_h
        )
        
        try:
            cropped = img.crop(crop_box)
            img_name = f"{SECTION_NAME}__--__{q['label']}__--__1.png"
            cropped.save(os.path.join(EXPORT_DIR, img_name))
        except Exception as e:
            log(f"❌ Crop Failed Q{q['label']}: {e}")

    # ZIP
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
    return "Team Stark V11 (Gemini 2.0 Integrated) 🚀"

@app.route('/process', methods=['POST'])
def upload_file():
    log("🔵 New Request Received")
    if 'pdf' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['pdf']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    
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
