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

# --- 🔑 CONFIG ---
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    log("⚠️ WARNING: GEMINI_API_KEY missing!")

# --- 🧠 AI ANALYSIS ---
def get_ai_config(image_paths):
    if not GENAI_API_KEY: return None
    try:
        log("🤖 Asking Gemini for Regex & Margins...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        content_parts = []
        for path in image_paths:
            content_parts.append(genai.upload_file(path))
        
        prompt = """
        Analyze these exam pages. Return ONLY a JSON object.
        1. "top_margin": Header height in pixels (e.g. 60).
        2. "bottom_margin": Footer height in pixels (e.g. 50).
        3. "regex_pattern": Python Regex to find Question Numbers at start of line.
           - Matches: "Q.1", "1.", "(1)", "Q1", "1 )"
           - STRICTLY NO equations or options.
        """
        content_parts.append(prompt)
        
        result = model.generate_content(content_parts)
        text_resp = result.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_resp)
        log(f"✅ AI Suggested: {data}")
        return data
    except Exception as e:
        log(f"❌ AI Failed: {e}")
        return None

# --- HELPER FUNCTIONS ---
def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_questions_with_strategy(doc, strategy_name, top_m, bot_m, regex_pattern):
    """
    Tries to extract questions using a specific strategy (Margins + Regex).
    Returns a list of questions found.
    """
    extracted_data = []
    log(f"🔄 Trying Strategy: {strategy_name} | Regex: {regex_pattern}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        height = page.rect.height
        mid_x = page.rect.width / 2
        blocks = page.get_text("blocks")
        
        for b in blocks:
            text = b[4].strip()
            bbox = b[:4]
            x0 = bbox[0]

            # Filters
            if bbox[1] < top_m or bbox[3] > height - bot_m: continue # Margins
            if x0 > 80 and x0 < mid_x: continue # Indentation Lock
            if x0 > mid_x + 80: continue
            if text.startswith("["): continue # Citations
            if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.I): continue

            try:
                # Regex Match
                q_match = re.search(regex_pattern, text, re.IGNORECASE)
                if q_match:
                    if any(x in text for x in ["Answer", "Solution", "Page", "Total"]): continue
                    
                    # Extract Number
                    q_no_str = None
                    for group in q_match.groups():
                        if group: 
                            q_no_str = group
                            break
                    
                    if not q_no_str: continue

                    # Validation
                    try:
                        q_val = int(q_no_str)
                        if q_val <= 0 or q_val > 500: continue
                    except: continue

                    extracted_data.append({
                        "label": q_no_str, 
                        "x0": bbox[0], "y0": bbox[1], 
                        "page": page_num, 
                        "bbox": bbox,
                        "text": text
                    })
            except: continue
            
    return extracted_data

def process_cbt_logic(pdf_path):
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    doc = fitz.open(pdf_path)
    
    # 1. Generate Images for AI
    pages_to_check = min(2, len(doc))
    img_paths = []
    for i in range(pages_to_check):
        pix = doc[i].get_pixmap(dpi=150)
        p_path = os.path.join(UPLOAD_FOLDER, f"analyze_page_{i}.jpg")
        pix.save(p_path)
        img_paths.append(p_path)

    # 2. DEFINE STRATEGIES (The Self-Healing Logic)
    strategies = []
    
    # Strategy A: AI (Priority 1)
    ai_data = get_ai_config(img_paths)
    if ai_data:
        strategies.append({
            "name": "GEMINI_AI",
            "top": int(ai_data.get("top_margin", 60)),
            "bot": int(ai_data.get("bottom_margin", 50)),
            "regex": ai_data.get("regex_pattern")
        })

    # Strategy B: Universal Fallback 1 (Standard Q.1)
    strategies.append({
        "name": "FALLBACK_STANDARD",
        "top": 60, "bot": 50,
        "regex": r"^(?:Q|Question|Que|No|Problem)?[\.\s\-]?\s*(\d+)[\.\)\-]"
    })
    
    # Strategy C: Aggressive Fallback (Loose matching for '1 .')
    strategies.append({
        "name": "FALLBACK_AGGRESSIVE",
        "top": 50, "bot": 50,
        "regex": r"^(\d+)\s*[\.\)]"
    })

    # 3. EXECUTE STRATEGIES
    final_questions = []
    used_strategy = "NONE"
    
    for strat in strategies:
        questions = extract_questions_with_strategy(doc, strat["name"], strat["top"], strat["bot"], strat["regex"])
        
        if len(questions) > 0:
            log(f"✅ Success! Found {len(questions)} questions using {strat['name']}")
            final_questions = questions
            used_strategy = strat["name"]
            break # Stop trying other strategies if one works
        else:
            log(f"⚠️ {strat['name']} found 0 questions. Trying next...")

    if not final_questions:
        raise Exception("Failed to crop ANY questions. Is this a Scanned PDF (Image-only)?")

    # 4. BUILD JSON & CROP IMAGES
    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": f"Team_Stark_{used_strategy}"
    }

    # Process discovered questions...
    for i, q in enumerate(final_questions):
        # Column Check
        page_width = doc[q["page"]].rect.width
        mid_x = page_width / 2
        is_right_col = q["x0"] > mid_x
        
        # Calculate Height
        # Simple Logic: Height is distance to next question OR arbitrary limit
        # Finding next question on same page/column
        next_q_y = doc[q["page"]].rect.height - 50 # Default to footer
        
        # Find strictly next question in list
        if i + 1 < len(final_questions):
            nq = final_questions[i+1]
            if nq["page"] == q["page"]:
                # Check column consistency
                nq_is_right = nq["x0"] > mid_x
                if is_right_col == nq_is_right:
                    next_q_y = nq["y0"] - 15

        height = next_q_y - q["y0"]
        if height < 20: height = 100 # Safety buffer

        # JSON Coords
        pg_h = doc[q["page"]].rect.height
        pg_w = doc[q["page"]].rect.width
        
        data_json["pdfCropperData"][SUBJECT_NAME][SECTION_NAME][str(q["label"])] = {
            "que": q["label"], "type": "mcq", "marks": {"cm": 4, "im": -1},
            "pdfData": [{
                "x1": 5 if not is_right_col else 505, 
                "x2": 495 if not is_right_col else 995, 
                "y1": round(((q["y0"] - 25)/pg_h)*1000), 
                "y2": round(((next_q_y + 10)/pg_h)*1000), 
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
        
        x1 = 0 if not is_right_col else mid_x
        x2 = mid_x if not is_right_col else pg_w
        
        crop_box = (
            x1 * scale_w, 
            max(0, q["y0"] - 30) * scale_h, 
            x2 * scale_w, 
            min(pg_h, next_q_y + 10) * scale_h
        )
        
        cropped = img.crop(crop_box)
        img_name = f"{SECTION_NAME}__--__{q['label']}__--__1.png"
        cropped.save(os.path.join(EXPORT_DIR, img_name))

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
    return "Team Stark Auto-Healing Backend V7 🚀"

@app.route('/process', methods=['POST'])
def upload_file():
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
