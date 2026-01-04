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
from groq import Groq

app = Flask(__name__)

# --- 🔥 SECURITY & CORS 🔥 ---
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

# --- ⚡ GROQ VISION CLIENT ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_ai_config(image_paths):
    if not client or not image_paths: return None
    
    # Using Page 2 (Index 1) usually avoids Cover Pages/Instructions
    target_img = image_paths[0] 
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    try:
        base64_img = encode_image(target_img)
        log(f"🤖 Groq Boss Analyzing Layout from {os.path.basename(target_img)}...")
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            You are the Layout Master. Look at this exam page.
                            Tell me EXACTLY how to crop it.
                            
                            Return ONLY a JSON object:
                            1. "top_margin": Int (Pixels to cut from top. e.g. 60).
                            2. "bottom_margin": Int (Pixels to cut from bottom. e.g. 50).
                            3. "left_margin": Int (Pixels to cut from left. e.g. 0).
                            4. "regex_pattern": Python Regex for Question Number.
                               - Example: "^Q\\\\.?[\\\\s-]?\\\\s*(\\\\d+)[\\\\.\\\\)]"
                               - ESCAPE ALL BACKSLASHES.
                            5. "is_two_column": Boolean (True if it looks like 2 columns).
                            6. "ignore_words": List[String] (Skip blocks containing these words e.g. ["Section A", "Rough Work"]).
                            
                            Output JSON only.
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1, 
            max_tokens=800,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        resp_content = completion.choices[0].message.content
        data = json.loads(resp_content)
        log(f"✅ AI Boss Orders: {data}")
        return data

    except Exception as e:
        log(f"❌ AI Failed: {str(e)}")
        return None

# --- HELPER FUNCTIONS ---
def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_questions_with_strategy(doc, strategy_name, config):
    extracted_data = []
    
    # --- STRICT OBEDIENCE TO AI ---
    top_m = int(config.get("top_margin", 50))
    bot_m = int(config.get("bottom_margin", 50))
    left_m = int(config.get("left_margin", 0))
    ai_regex = config.get("regex_pattern", "")
    ignore_words = config.get("ignore_words", [])
    
    # Priority Regex List (AI First)
    regex_list = []
    if ai_regex: regex_list.append(ai_regex)
    # Backups just in case AI gives invalid regex
    regex_list.append(r"^(?:Q|Question|Que|No|Problem)?[\.\s\-]?\s*(\d+)[\.\)\-]")
    regex_list.append(r"^(\d+)\s*[\.\)]")

    log(f"🔄 Processing {len(doc)} pages | AI Margins: T{top_m}/B{bot_m}/L{left_m}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        height = page.rect.height
        blocks = page.get_text("blocks")
        
        for b in blocks:
            text = b[4].strip()
            bbox = b[:4]
            x0, y0, x1, y1 = bbox

            # 1. AI Margin Check
            if y0 < top_m or y1 > height - bot_m: continue
            if x0 < left_m: continue
            
            # 2. AI Ignore Words Check
            if any(w.lower() in text.lower() for w in ignore_words): continue
            
            # 3. Basic Garbage Filter
            if text.startswith("[") or re.search(r"@[a-z]+\.", text, re.I): continue
            if any(x in text for x in ["Answer", "Solution", "Sol.", "Ans."]): continue

            # 4. Regex Match
            for pat in regex_list:
                try:
                    q_match = re.search(pat, text, re.IGNORECASE)
                    if q_match:
                        q_no_str = next((g for g in q_match.groups() if g), None)
                        if not q_no_str: continue
                        
                        try:
                            q_val = int(q_no_str)
                            if q_val <= 0 or q_val > 500: continue
                        except: continue

                        extracted_data.append({
                            "label": q_no_str, 
                            "x0": x0, "y0": y0, 
                            "page": page_num, 
                            "bbox": bbox
                        })
                        break 
                except: continue
            
    return extracted_data

def process_cbt_logic(pdf_path):
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    doc = fitz.open(pdf_path)
    
    # --- INTELLIGENT SAMPLING ---
    # Try to scan Page 2 (index 1) to avoid Cover Page bias.
    # If 1 page PDF, scan Page 1 (index 0).
    target_page_idx = 1 if len(doc) > 1 else 0
    
    pix = doc[target_page_idx].get_pixmap(dpi=150)
    p_path = os.path.join(UPLOAD_FOLDER, "analyze_target.jpg")
    pix.save(p_path)
    img_paths = [p_path]

    # Ask Groq
    ai_data = get_ai_config(img_paths)
    
    config = {
        "top_margin": 50, "bottom_margin": 50, "left_margin": 0,
        "is_two_column": False, "regex_pattern": "", "ignore_words": []
    }
    
    strategy_name = "FALLBACK_STD"
    if ai_data:
        config.update(ai_data)
        strategy_name = "GROQ_MASTER"

    # Extract
    final_questions = extract_questions_with_strategy(doc, strategy_name, config)

    # Panic Mode
    if len(final_questions) == 0:
        log("⚠️ AI Margins too strict? Retrying with Zero Margins...")
        config["top_margin"] = 0
        config["bottom_margin"] = 0
        config["left_margin"] = 0
        final_questions = extract_questions_with_strategy(doc, "PANIC_MODE", config)

    if not final_questions:
        raise Exception("Failed to crop ANY questions.")

    log(f"✅ Total Qs: {len(final_questions)}")

    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": f"Team_Stark_{strategy_name}"
    }

    # AI Dictated Column Logic
    force_two_col = config.get("is_two_column", False)

    for i, q in enumerate(final_questions):
        pg_h = doc[q["page"]].rect.height
        pg_w = doc[q["page"]].rect.width
        mid_x = pg_w / 2
        
        is_right_col = q["x0"] > mid_x
        
        if force_two_col or is_right_col:
            x1 = 0 if not is_right_col else mid_x
            x2 = mid_x if not is_right_col else pg_w
            json_x1 = 5 if not is_right_col else 505
            json_x2 = 495 if not is_right_col else 995
        else:
            x1 = 0
            x2 = pg_w
            json_x1 = 5
            json_x2 = 995

        next_q_y = pg_h - 50 
        
        if i + 1 < len(final_questions):
            nq = final_questions[i+1]
            if nq["page"] == q["page"]:
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
    return "Team Stark V22 (AI Boss Mode) 🚀"

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
