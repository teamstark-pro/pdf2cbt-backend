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
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

EXPORT_DIR = "/tmp/cbt_master_package"
UPLOAD_FOLDER = "/tmp/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 🔑 GEMINI API CONFIGURATION ---
# Railway ke "Variables" tab mein GEMINI_API_KEY daalna mat bhoolna!
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables!")

# --- 🧠 AI ANALYSIS FUNCTION ---
def analyze_page_with_gemini(image_path):
    """
    Sends the first page to Gemini 1.5 Flash to detect Margins & Regex.
    """
    if not GENAI_API_KEY:
        print("❌ No API Key. Using Default Fallback.")
        return None

    try:
        print("🤖 Asking Gemini to analyze layout...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Upload image to Gemini
        myfile = genai.upload_file(image_path)
        
        prompt = """
        Analyze this exam page image. I need to crop questions.
        Return ONLY a raw JSON object (no markdown, no backticks).
        Fields required:
        1. "top_margin": Height in pixels of the header area to ignore (e.g., 60).
        2. "bottom_margin": Height in pixels of the footer area to ignore (e.g., 50).
        3. "regex_pattern": A Python Regex string to capture the Question Number at the start of a line.
           - If text is "Q.1 ...", return "^Q\\.?[\\s-]?\\s?(\\d+)[\\.\\)]"
           - If text is "1. ...", return "^(\\d+)[\\.\\)]"
           - If text is "(1) ...", return "^\\((\\d+)\\)"
        
        Be precise. Default to top_margin: 60, bottom_margin: 50 if unsure.
        """
        
        result = model.generate_content([myfile, prompt])
        
        # Clean response
        text_resp = result.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_resp)
        print(f"✅ Gemini Analysis: {data}")
        return data

    except Exception as e:
        print(f"⚠️ Gemini Failed: {e}. Switching to Default Logic.")
        return None

# --- HELPER FUNCTIONS ---

def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def process_cbt_logic(pdf_path):
    # 1. CLEANUP
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    # 2. GENERATE PAGE 1 IMAGE FOR AI
    doc = fitz.open(pdf_path)
    page1 = doc[0]
    pix = page1.get_pixmap(dpi=150) # Faster upload
    img_path = os.path.join(UPLOAD_FOLDER, "analyze_page.jpg")
    pix.save(img_path)
    
    # 3. GET PARAMETERS (AI vs DEFAULT)
    ai_config = analyze_page_with_gemini(img_path)
    
    if ai_config:
        TOP_MARGIN = int(ai_config.get("top_margin", 60))
        BOTTOM_MARGIN = int(ai_config.get("bottom_margin", 50))
        # Regex string from JSON needs careful handling
        MASTER_REGEX = ai_config.get("regex_pattern", r"^(?:(?:Q|Question|Que|No)[\.\s\-]?\s?(\d+)|(\d+)[\.\)\-])")
    else:
        # Fallback Defaults (The "God Mode" logic)
        TOP_MARGIN = 60
        BOTTOM_MARGIN = 50
        MASTER_REGEX = r"^(?:(?:Q|Question|Que|Problem|No)[\.\s\-]?\s?(\d+)|(\d+)[\.\)\-])"

    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": "Team_Stark_AI_Integrated"
    }
    
    print(f"🚀 Processing {len(doc)} pages using: Top={TOP_MARGIN}, Bot={BOTTOM_MARGIN}, Regex={MASTER_REGEX}")

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
            
            # --- FILTER 1: AI MARGINS ---
            if bbox[1] < TOP_MARGIN or bbox[3] > height - BOTTOM_MARGIN:
                continue

            # --- FILTER 2: X-AXIS LOCK (Safety Net) ---
            # Even with AI, we don't want side equations
            if x0 > 70 and x0 < mid_x: continue
            if x0 > mid_x + 70: continue

            # --- FILTER 3: COMMON NOISE ---
            if text.startswith("["): continue # Citations
            if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.I): continue # Emails

            try:
                q_match = re.search(MASTER_REGEX, text, re.IGNORECASE)
                if q_match:
                    if any(x in text for x in ["Answer", "Solution", "Page", "Total", "Notes"]): continue
                    
                    # Extract Number
                    q_no_str = None
                    for group in q_match.groups():
                        if group:
                            q_no_str = group
                            break
                    
                    if not q_no_str: continue

                    # Range Check
                    try:
                        q_val = int(q_no_str)
                        if q_val <= 0 or q_val > 500: continue
                    except:
                        continue

                    all_q.append({"label": q_no_str, "x0": bbox[0], "y0": bbox[1]})
                    
                    if bbox[0] > mid_x + 30: is_multi_column = True
            except Exception as e:
                # If AI regex fails, skip block
                continue

        if not all_q: continue

        # Sort & Columns
        if is_multi_column:
            left_col = sorted([q for q in all_q if q['x0'] < mid_x], key=lambda x: x['y0'])
            right_col = sorted([q for q in all_q if q['x0'] >= mid_x], key=lambda x: x['y0'])
            columns = [(left_col, 0, mid_x), (right_col, mid_x, width)]
        else:
            columns = [(sorted(all_q, key=lambda x: x['y0']), 0, width)]

        # Crop & Json Logic
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

    # ZIP Creation
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
    return "Team Stark AI Backend is LIVE! 🚀"

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
            print(f"ERROR: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
