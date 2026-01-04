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

app = Flask(__name__)
CORS(app)

EXPORT_DIR = "/tmp/cbt_master_package"
UPLOAD_FOLDER = "/tmp/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- HELPER FUNCTIONS ---

def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def process_cbt_logic(pdf_path):
    # 1. CLEANUP & SETUP
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # --- NAME CHANGE: STARK ---
    SECTION_NAME = "Stark"
    SUBJECT_NAME = "Stark"
    
    # JSON Config
    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": "Team_Stark_IntelliCropper_V2"
    }
    
    doc = fitz.open(pdf_path)
    
    # 2. IMPROVED REGEX (STRICTER)
    # Explanation:
    # ^             -> Start of line
    # (Q...)?       -> Optional 'Q', 'Question', etc.
    # [\.\s\-]?     -> Optional separator (dot, space, dash)
    # (\d+)         -> THE NUMBER (Capture Group 2)
    # [\.\)\-]      -> MUST be followed by dot, bracket, or dash (No space allowed to avoid random numbers)
    MASTER_REGEX = r"^(Q|Question|Que|Problem|Prob|No|S)?[\.\s\-]?\s?(\d+)[\.\)\-]"
    
    print(f"Processing {len(doc)} pages with STARK Logic...")

    for page_num in range(len(doc)):
        page = doc[page_num]
        width, height = page.rect.width, page.rect.height
        mid_x = width / 2
        
        # Get text blocks (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        
        all_q = []
        is_multi_column = False

        # 3. SMART SCANNING
        for b in blocks:
            text = b[4].strip()
            bbox = b[:4] # [x0, y0, x1, y1]
            
            # --- FILTER: IGNORE HEADERS/FOOTERS ---
            # Agar text page ke top 50px ya bottom 50px mein hai, to ignore karo.
            if bbox[1] < 50 or bbox[3] > height - 50:
                continue

            # Anti-Noise (Email/Phone)
            if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.I): continue

            # Regex Match
            q_match = re.match(MASTER_REGEX, text, re.IGNORECASE)
            if q_match:
                # Anti-Solution/Header words check
                if any(x in text for x in ["Answer", "Solution", "Page", "Total", "Notes", "Marks"]): continue
                
                # Extract Number specifically from Group 2
                q_no_raw = q_match.group(2) 
                
                # Valid Question Found!
                all_q.append({"label": q_no_raw, "x0": bbox[0], "y0": bbox[1]})
                
                # Check for 2-Column Layout
                if bbox[0] > mid_x + 30: is_multi_column = True

        if not all_q: 
            print(f"Skipping Page {page_num+1} (No Questions Found)")
            continue

        # 4. SORTING & COLUMNS
        if is_multi_column:
            left_col = sorted([q for q in all_q if q['x0'] < mid_x], key=lambda x: x['y0'])
            right_col = sorted([q for q in all_q if q['x0'] >= mid_x], key=lambda x: x['y0'])
            columns = [(left_col, 0, mid_x), (right_col, mid_x, width)]
        else:
            columns = [(sorted(all_q, key=lambda x: x['y0']), 0, width)]

        # 5. CROPPING LOGIC
        for col_questions, start_x, end_x in columns:
            for i, q in enumerate(col_questions):
                q_id = q["label"]
                top_y = q["y0"]
                
                # Determine Bottom Y (Next question's top OR Page bottom)
                if i + 1 < len(col_questions):
                    bottom_y = col_questions[i+1]["y0"] - 15 # Gap before next Q
                else:
                    bottom_y = height - 50 # Page footer margin

                # SAFETY CHECK: If crop height is too small (garbage detection), skip
                if (bottom_y - top_y) < 20: 
                    continue

                # JSON Coordinates (0-1000 scale)
                # Adding slight padding (-25 top, +10 bottom) to catch full text
                json_y1 = round(((max(0, top_y - 25)) / height) * 1000)
                json_y2 = round(((min(height, bottom_y + 10)) / height) * 1000)
                json_x1 = round((start_x / width) * 1000)
                json_x2 = round((end_x / width) * 1000)

                # Add to JSON
                data_json["pdfCropperData"][SUBJECT_NAME][SECTION_NAME][str(q_id)] = {
                    "que": q_id, "type": "mcq", "marks": {"cm": 4, "im": -1},
                    "pdfData": [{
                        "x1": max(5, json_x1), 
                        "x2": min(995, json_x2), 
                        "y1": max(0, json_y1), 
                        "y2": min(1000, json_y2), 
                        "page": page_num + 1
                    }],
                    "answerOptions": "4"
                }

                # IMAGE GENERATION (High Quality)
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                scale_h = pix.height / height
                scale_w = pix.width / width
                
                # Apply padding for Image Crop
                crop_top = max(0, top_y - 30) * scale_h
                crop_bottom = min(height, bottom_y + 10) * scale_h
                crop_left = start_x * scale_w
                crop_right = end_x * scale_w
                
                cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
                
                # Save Image with STARK Name
                img_name = f"{SECTION_NAME}__--__{q_id}__--__1.png"
                cropped.save(os.path.join(EXPORT_DIR, img_name))

    # 6. FINALIZE ZIP
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
    return "Team Stark Backend is LIVE! 🚀"

@app.route('/process', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        try:
            zip_buffer = process_cbt_logic(filepath)
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='TeamStark_Result.zip'
            )
        except Exception as e:
            # Print error to Railway Logs for debugging
            print(f"ERROR: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
