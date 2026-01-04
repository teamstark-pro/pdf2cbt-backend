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
import sys # For flushing logs
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

EXPORT_DIR = "/tmp/cbt_master_package"
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 🔍 LOGGING HELPER ---
def log(msg):
    """Forces print to show up in Railway logs immediately."""
    print(f"[STARK LOG] {msg}", file=sys.stdout, flush=True)

# --- 🔑 CONFIG ---
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    log(f"✅ API Key Loaded: {GENAI_API_KEY[:5]}...******")
else:
    log("❌ ERROR: GEMINI_API_KEY missing in Railway Variables!")

# --- 🧠 DUAL PAGE AI ANALYSIS ---
def analyze_with_gemini_dual_page(image_paths):
    """
    Sends up to 2 pages to Gemini to ignore cover sheets and find real questions.
    """
    if not GENAI_API_KEY:
        log("⚠️ Skipping AI: No Key.")
        return None

    try:
        log("🤖 Initiating Gemini 1.5 Flash Session...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare content list (Prompt + Images)
        content_parts = []
        
        # Add Images
        for path in image_paths:
            log(f"📤 Uploading Image to AI: {os.path.basename(path)}")
            file_ref = genai.upload_file(path)
            content_parts.append(file_ref)
        
        # The Prompt
        prompt = """
        I have provided images of the first 1-2 pages of an exam PDF.
        Page 1 might be a cover sheet/instructions. Look at BOTH pages to find the actual question format.
        
        I need to crop questions using Python Regex. Return ONLY a JSON object:
        1. "top_margin": Height in pixels of header to ignore (e.g. 60).
        2. "bottom_margin": Height in pixels of footer to ignore (e.g. 50).
        3. "regex_pattern": The Python Regex to capture the Question Number at start of a line.
           - If questions look like "Q.1", return "^Q\\.?[\\s-]?\\s?(\\d+)[\\.\\)]"
           - If questions look like "1 .", return "^(\\d+)\\s*[\\.]"
           - If questions look like "(1)", return "^\\((\\d+)\\)"
           - If questions look like "Q1", return "^Q\\s?(\\d+)"
        
        IMPORTANT: Do not match equation numbers like "eq(1)" or option numbers like "(1)". match strictly START of line labels.
        """
        content_parts.append(prompt)

        log("🚀 Sending Request to Gemini...")
        result = model.generate_content(content_parts)
        
        # Parse Response
        raw_text = result.text
        log(f"📥 Raw Gemini Response: {raw_text[:100]}...") # Print first 100 chars
        
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        log(f"✅ Parsed AI Config: {data}")
        return data

    except Exception as e:
        log(f"❌ Gemini Crash: {str(e)}")
        return None

# --- HELPER FUNCTIONS ---
def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def process_cbt_logic(pdf_path):
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    # 1. PREPARE IMAGES (First 2 Pages)
    doc = fitz.open(pdf_path)
    pages_to_check = min(2, len(doc))
    img_paths = []
    
    log(f"📸 Generating images for first {pages_to_check} pages...")
    
    for i in range(pages_to_check):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        p_path = os.path.join(UPLOAD_FOLDER, f"analyze_page_{i}.jpg")
        pix.save(p_path)
        img_paths.append(p_path)
    
    # 2. ASK AI
    ai_config = analyze_with_gemini_dual_page(img_paths)
    
    # 3. SET CONFIG
    if ai_config:
        TOP_MARGIN = int(ai_config.get("top_margin", 60))
        BOTTOM_MARGIN = int(ai_config.get("bottom_margin", 50))
        MASTER_REGEX = ai_config.get("regex_pattern")
        SOURCE = "GEMINI_AI"
    else:
        log("⚠️ Using Fallback Config.")
        TOP_MARGIN = 60
        BOTTOM_MARGIN = 50
        MASTER_REGEX = r"^(?:(?:Q|Question|Que|No|Problem)?[\.\s\-]?\s*(\d+)[\.\)\-])"
        SOURCE = "FALLBACK_LOGIC"

    log(f"⚙️ FINAL CONFIG | Source: {SOURCE} | Regex: {MASTER_REGEX} | Top: {TOP_MARGIN}")

    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": f"Team_Stark_{SOURCE}"
    }

    # 4. PROCESS PAGES
    total_q_found = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        width, height = page.rect.width, page.rect.height
        mid_x = width / 2
        blocks = page.get_text("blocks")
        
        all_q = []
        is_multi_column = False

        for b in blocks:
            text = b[4].strip()
            bbox = b[:4]
            x0 = bbox[0]
            
            if bbox[1] < TOP_MARGIN or bbox[3] > height - BOTTOM_MARGIN: continue
            
            # Strict X-Axis Lock (Even with AI, prevent side-notes)
            if x0 > 80 and x0 < mid_x: continue
            if x0 > mid_x + 80: continue

            if text.startswith("["): continue 
            if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.I): continue

            try:
                q_match = re.search(MASTER_REGEX, text, re.IGNORECASE)
                if q_match:
                    if any(x in text for x in ["Answer", "Solution", "Page", "Total"]): continue
                    
                    q_no_str = None
                    for group in q_match.groups():
                        if group: 
                            q_no_str = group
                            break
                    
                    if not q_no_str: continue

                    try:
                        q_val = int(q_no_str)
                        if q_val <= 0 or q_val > 500: continue
                    except: continue

                    all_q.append({"label": q_no_str, "x0": bbox[0], "y0": bbox[1]})
                    if bbox[0] > mid_x + 30: is_multi_column = True
            except: continue

        if not all_q: 
            # log(f"Page {page_num+1}: 0 Questions.") # Commented to reduce noise
            continue
        
        total_q_found += len(all_q)

        # Columns & Crop Logic...
        if is_multi_column:
            left_col = sorted([q for q in all_q if q['x0'] < mid_x], key=lambda x: x['y0'])
            right_col = sorted([q for q in all_q if q['x0'] >= mid_x], key=lambda x: x['y0'])
            columns = [(left_col, 0, mid_x), (right_col, mid_x, width)]
        else:
            columns = [(sorted(all_q, key=lambda x: x['y0']), 0, width)]

        for col_questions, start_x, end_x in columns:
            for i, q in enumerate(col_questions):
                q_id = q["label"]
                top_y = q["y0"]
                
                if i + 1 < len(col_questions):
                    bottom_y = col_questions[i+1]["y0"] - 15
                else:
                    bottom_y = height - BOTTOM_MARGIN

                if (bottom_y - top_y) < 20: continue

                json_y1 = round(((max(0, top_y - 25)) / height) * 1000)
                json_y2 = round(((min(height, bottom_y + 10)) / height) * 1000)
                json_x1 = round((start_x / width) * 1000)
                json_x2 = round((end_x / width) * 1000)

                data_json["pdfCropperData"][SUBJECT_NAME][SECTION_NAME][str(q_id)] = {
                    "que": q_id, "type": "mcq", "marks": {"cm": 4, "im": -1},
                    "pdfData": [{"x1": max(5, json_x1), "x2": min(995, json_x2), "y1": max(0, json_y1), "y2": min(1000, json_y2), "page": page_num + 1}],
                    "answerOptions": "4"
                }

                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                scale_h = pix.height / height
                scale_w = pix.width / width
                
                crop_box = (
                    start_x * scale_w, 
                    max(0, top_y - 30) * scale_h, 
                    end_x * scale_w, 
                    min(height, bottom_y + 10) * scale_h
                )
                
                cropped = img.crop(crop_box)
                img_name = f"{SECTION_NAME}__--__{q_id}__--__1.png"
                cropped.save(os.path.join(EXPORT_DIR, img_name))

    log(f"🏁 Processing Complete. Total Questions: {total_q_found}")
    
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
    return "Stark Backend with Live Logs 🚀"

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
            log(f"🔥 CRITICAL ERROR: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
