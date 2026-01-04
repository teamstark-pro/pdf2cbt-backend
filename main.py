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
# 1500 RPD Free Tier Key (AI Studio)
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

# --- 🧠 AI ANALYSIS WITH VALIDATION ---
def get_layout_config(doc, image_path):
    """
    1. Asks AI for config.
    2. Tests AI config on Page 1 text.
    3. If AI fails, returns Default Config.
    """
    
    # --- DEFAULT CONFIG (The "Desi" Logic) ---
    # Regex logic: 
    # ^\s* -> Line start (allow spaces)
    # (?:Q...)?  -> Optional 'Q', 'Question', 'No'
    # [\.\s]?    -> Optional dot/space separator
    # (\d+)      -> THE NUMBER (Capture Group)
    # [\.\)\-]   -> Must end with dot, bracket, or dash
    default_config = {
        "top_margin": 50,
        "bottom_margin": 50,
        "regex": r"^\s*(?:Q|Question|Que|No|Problem)?[\.\s\-]?\s*(\d+)[\.\)\-]",
        "source": "DEFAULT_FALLBACK"
    }

    if not GENAI_API_KEY:
        print("⚠️ No API Key. Using Default.")
        return default_config

    try:
        # 1. Ask Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        myfile = genai.upload_file(image_path)
        
        prompt = """
        Analyze this exam page. I need to crop questions via Python Regex.
        Return ONLY a JSON object.
        1. "top_margin": Header height in pixels (approx).
        2. "bottom_margin": Footer height in pixels (approx).
        3. "regex_pattern": Python Regex to capture the Question Number at the start of a line.
           Examples:
           - "Q.1" -> "^Q\\.?\\s*(\\d+)[\\.\\)]"
           - "1 ." -> "^(\\d+)\\s*[\\.]"
           - "(1)" -> "^\\((\\d+)\\)"
        """
        
        result = model.generate_content([myfile, prompt])
        text_resp = result.text.replace("```json", "").replace("```", "").strip()
        ai_data = json.loads(text_resp)
        
        ai_regex = ai_data.get("regex_pattern", "")
        
        # 2. VALIDATION STEP (Crucial!)
        # Check if AI regex actually finds anything on Page 1
        page1_text = doc[0].get_text()
        matches = re.findall(ai_regex, page1_text, re.MULTILINE)
        
        if len(matches) > 0:
            print(f"✅ AI Regex Validated! Found {len(matches)} matches on Page 1.")
            return {
                "top_margin": int(ai_data.get("top_margin", 50)),
                "bottom_margin": int(ai_data.get("bottom_margin", 50)),
                "regex": ai_regex,
                "source": "GEMINI_AI"
            }
        else:
            print(f"❌ AI Regex '{ai_regex}' found 0 matches. Reverting to Default.")
            return default_config

    except Exception as e:
        print(f"⚠️ AI Error: {e}. Using Default.")
        return default_config

# --- HELPER FUNCTIONS ---

def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def process_cbt_logic(pdf_path):
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    # Load PDF
    doc = fitz.open(pdf_path)
    
    # Generate Page 1 Image for AI
    page1 = doc[0]
    pix = page1.get_pixmap(dpi=150)
    img_path = os.path.join(UPLOAD_FOLDER, "analyze_page.jpg")
    pix.save(img_path)
    
    # GET SMART CONFIG (AI or Fallback)
    config = get_layout_config(doc, img_path)
    
    TOP_MARGIN = config["top_margin"]
    BOTTOM_MARGIN = config["bottom_margin"]
    MASTER_REGEX = config["regex"]
    
    print(f"🚀 Processing with [{config['source']}] | Regex: {MASTER_REGEX}")

    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": f"Team_Stark_{config['source']}"
    }

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
            
            # --- FILTER 1: MARGINS ---
            if bbox[1] < TOP_MARGIN or bbox[3] > height - BOTTOM_MARGIN:
                continue

            # --- FILTER 2: X-AXIS LOCK (Safety Net) ---
            # If text is too far right (indented), it's likely an equation/option
            if x0 > 80 and x0 < mid_x: continue
            if x0 > mid_x + 80: continue

            # --- FILTER 3: COMMON NOISE ---
            if text.startswith("["): continue 
            if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.I): continue

            try:
                # Use the Selected Regex
                q_match = re.search(MASTER_REGEX, text, re.IGNORECASE)
                
                if q_match:
                    if any(x in text for x in ["Answer", "Solution", "Page", "Total", "Notes"]): continue
                    
                    # Extract Number safely
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
            except:
                continue

        if not all_q: 
            print(f"Page {page_num+1}: No questions found.")
            continue

        # Sort & Columns logic
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
    return "Team Stark Backend (Fail-Safe) is LIVE! 🚀"

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
