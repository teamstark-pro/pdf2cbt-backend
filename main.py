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
CORS(app)  # Allow Frontend to talk to Backend

EXPORT_DIR = "/tmp/cbt_master_package"
UPLOAD_FOLDER = "/tmp/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- YOUR ORIGINAL LOGIC FUNCTIONS ---

def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def process_cbt_logic(pdf_path):
    # Clean previous run
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    SECTION_NAME = "Mathematics Section 1"
    SUBJECT_NAME = "Mathematics"
    
    # JSON Config
    data_json = {
        "testConfig": {"pdfFileHash": get_pdf_hash(pdf_path)},
        "pdfCropperData": {SUBJECT_NAME: {SECTION_NAME: {}}},
        "appVersion": "1.30.0",
        "generatedBy": "Master_Universal_IntelliCropper_Railway_V1"
    }
    
    doc = fitz.open(pdf_path)
    MASTER_REGEX = r"^(Q|Question|Que|Problem|Prob|No|S)?[\.\s\-]?\s?(\d+|[IVXLC]+|[a-e])[\.\)\-\s]"
    
    print(f"Processing {len(doc)} pages...")

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
            
            # Anti-Noise
            if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.I) or re.search(r"\d{10,}", text):
                continue

            q_match = re.match(MASTER_REGEX, text, re.IGNORECASE)
            if q_match:
                if any(x in text for x in ["Answer", "Solution", "Page", "Total", "Notes"]): continue
                
                q_label = q_match.group(0).strip().rstrip('.)- ')
                try:
                    q_no_raw = re.search(r"(\d+|[IVXLC]+|[a-e])", q_label, re.I).group()
                except:
                    q_no_raw = q_label

                all_q.append({"label": q_no_raw, "x0": bbox[0], "y0": bbox[1]})
                if bbox[0] > mid_x + 30: is_multi_column = True

        if not all_q: continue

        # Sort Logic
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
                bottom_y = col_questions[i+1]["y0"] - 10 if i + 1 < len(col_questions) else height - 40

                json_y1 = round(((top_y - 20) / height) * 1000)
                json_y2 = round(((bottom_y + 10) / height) * 1000)
                json_x1 = round((start_x / width) * 1000)
                json_x2 = round((end_x / width) * 1000)

                data_json["pdfCropperData"][SUBJECT_NAME][SECTION_NAME][str(q_id)] = {
                    "que": q_id, "type": "mcq", "marks": {"cm": 4, "im": -1},
                    "pdfData": [{"x1": max(5, json_x1), "x2": min(995, json_x2), "y1": max(0, json_y1), "y2": min(1000, json_y2), "page": page_num + 1}],
                    "answerOptions": "4"
                }

                # Image Save
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                scale_h, scale_w = pix.height / height, pix.width / width
                
                crop_box = (start_x * scale_w, (top_y - 25) * scale_h, end_x * scale_w, (bottom_y + 15) * scale_h)
                cropped = img.crop(crop_box)
                img_name = f"{SECTION_NAME}__--__{q_id}__--__1.png"
                cropped.save(os.path.join(EXPORT_DIR, img_name))

    # Save JSON
    with open(os.path.join(EXPORT_DIR, "data.json"), "w") as f:
        json.dump(data_json, f, indent=2)

    # ZIP Creation
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
    return "Team Stark Backend is Running! 🚀"

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
            # Run the GOD SCRIPT
            zip_buffer = process_cbt_logic(filepath)
            
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='TeamStark_Result.zip'
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
