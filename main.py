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
import threading
import uuid
import random
from groq import Groq

app = Flask(__name__)

# --- 🔥 CORS & CONFIG 🔥 ---
CORS(app, resources={r"/*": {"origins": "*"}}, 
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["*"],
     expose_headers=["Content-Disposition", "Content-Type"])

EXPORT_DIR = "/tmp/cbt_master_package"
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global Job Store
jobs = {}

# --- 🔐 SECURITY ---
STARK_SECRET = os.environ.get("STARK_SECRET_KEY", "open_access_mode")
def is_authorized(req):
    if STARK_SECRET == "open_access_mode": return True
    client_key = req.headers.get("x-stark-secret")
    return client_key == STARK_SECRET

# --- ⚡ GROQ CLIENT ---
RAW_KEYS = os.environ.get("GROQ_API_KEYS", "")
API_KEY_POOL = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

def get_groq_client():
    if not API_KEY_POOL: return None
    return Groq(api_key=random.choice(API_KEY_POOL))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def job_log(job_id, msg):
    print(f"[{job_id}] {msg}", file=sys.stdout, flush=True)
    if job_id in jobs:
        timestamp = time.strftime("%H:%M:%S")
        jobs[job_id]["logs"].append(f"[{timestamp}] {msg}")

# --- 🤖 GROQ SCANNER ---
def get_questions_batch(job_id, image_paths):
    client = get_groq_client()
    if not client: return None
    
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    try:
        message_content = [
            {
                "type": "text",
                "text": """
                You are a Question Detection System. Scan these exam pages.
                
                MISSION:
                Identify every single Question Number that starts a question block.
                
                PATTERNS TO CATCH:
                - Standard: "1.", "Q1", "Q.1", "Q-1"
                - Brackets: "(1)", "1)"
                - Standalone: "1" (if it looks like a label)
                
                STRICT EXCLUSIONS:
                - DO NOT include numbers inside "Solution", "Ans", "Hint", "Explanation".
                - DO NOT include Page Numbers.
                
                OUTPUT:
                Return JSON mapping image index to a LIST of INTEGERS.
                Example: { "img_0": [1, 2, 3], "img_1": [4, 5] }
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
            temperature=0.1, max_tokens=800, top_p=1, stream=False,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        return data
    except Exception as e:
        job_log(job_id, f"❌ Groq Error: {str(e)}")
        return None

# --- ✂️ PROCESSOR ---
def extract_questions(job_id, doc, page_map):
    # Flatten map -> [(page, number)]
    all_targets = []
    for p, qs in page_map.items():
        for q in qs:
            all_targets.append({"page": p, "val": q})
    
    # Regexes (Broad to Narrow)
    # 1. Explicit labels (Q.1, Q1)
    # 2. Start of line numbers (1., 1))
    # 3. Standalone numbers
    regex_list = [
        r"^\s*(?:Q|Question|Que|No)[\.\s\-]?\s*(\d+)", 
        r"^\s*(\d+)[\.\)\-\:]", 
        r"^\s*(\d+)\s*$" 
    ]
    
    BAD_KEYWORDS = ["Solution", "Detailed Solution", "Correct Answer", "Explanation", "Ans."]
    
    valid_coords = []
    
    # 1. FIND LOCATIONS
    for item in all_targets:
        p_idx = item['page']
        q_target = item['val']
        page = doc[p_idx]
        blocks = page.get_text("blocks")
        
        found = False
        for b in blocks:
            text = b[4].strip()
            if not text: continue
            
            # Anti-Solution Shield
            if any(bad in text for bad in BAD_KEYWORDS): continue
            
            # Regex Match
            for pat in regex_list:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        if val == q_target:
                            valid_coords.append({
                                "label": str(val),
                                "page": p_idx,
                                "y0": b[1],
                                "y1": b[3]
                            })
                            found = True
                            break
                    except: continue
            if found: break
    
    # Sort Spatially: Page -> Y Position
    valid_coords.sort(key=lambda x: (x["page"], x["y0"]))
    job_log(job_id, f"✅ Verified {len(valid_coords)} locations via OCR match.")
    
    # 2. CROP & GENERATE JSON DATA
    final_json_data = {}
    
    for i, q in enumerate(valid_coords):
        curr_p = q["page"]
        
        # Calculate Crop Box
        # Start: slightly above found label
        y_start = max(0, q["y0"] - 10)
        
        # End: find next question or end of page
        pg_h = doc[curr_p].rect.height
        pg_w = doc[curr_p].rect.width
        
        if i + 1 < len(valid_coords):
            next_q = valid_coords[i+1]
            if next_q["page"] == curr_p:
                y_end = next_q["y0"] - 15
            else:
                y_end = pg_h - 50 # End of current page
        else:
            y_end = pg_h - 50 # Last question
            
        # Ensure min height
        if y_end <= y_start: y_end = y_start + 100
        
        # --- CROP IMAGE ---
        page = doc[curr_p]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        scale_w = pix.width / pg_w
        scale_h = pix.height / pg_h
        
        # Full width crop (0 to pg_w)
        crop_box = (0, y_start * scale_h, pg_w * scale_w, y_end * scale_h)
        
        try:
            cropped = img.crop(crop_box)
            img_name = f"Stark__--__{q['label']}__--__1.png"
            
            # Handle duplicates (e.g. Q1 in Section A, Q1 in Section B)
            if os.path.exists(os.path.join(EXPORT_DIR, img_name)):
                 img_name = f"Stark__--__{q['label']}_p{curr_p}__--__1.png"
            
            cropped.save(os.path.join(EXPORT_DIR, img_name))
            
            # --- JSON ENTRY (FIXED STRUCTURE) ---
            # Using str(q['label']) as key.
            # If duplicate, we use unique key but keep "que" label same
            unique_key = q['label']
            if "_" in img_name and "_p" in img_name:
                unique_key = f"{q['label']}_{curr_p}"

            final_json_data[unique_key] = {
                "que": q['label'],
                "type": "mcq",
                "marks": {"cm": 4, "im": -1},
                "pdfData": [{
                    "x1": 5, 
                    "x2": 995, 
                    "y1": round((y_start/pg_h)*1000), 
                    "y2": round((y_end/pg_h)*1000), 
                    "page": curr_p + 1 # 1-based index
                }],
                "answerOptions": "4"
            }
            
        except Exception as e:
            job_log(job_id, f"⚠️ Crop failed for Q{q['label']}: {e}")
            
    return final_json_data

# --- 🧵 WORKER THREAD ---
def process_thread(job_id, pdf_path):
    try:
        job_log(job_id, "🚀 Initialization...")
        if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        pdf_hash = get_pdf_hash(pdf_path) # Hash for JSON
        
        FULL_MAP = {}
        BATCH_SIZE = 2
        
        # PHASE 1: SCAN
        for i in range(0, len(doc), BATCH_SIZE):
            indices = range(i, min(i + BATCH_SIZE, len(doc)))
            img_paths = []
            
            job_log(job_id, f"🤖 Scanning Pages {[x+1 for x in indices]}...")
            for p in indices:
                pix = doc[p].get_pixmap(dpi=100)
                p_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{p}.jpg")
                pix.save(p_path)
                img_paths.append(p_path)
            
            res = get_questions_batch(job_id, img_paths)
            if res:
                for idx, key in enumerate(sorted(res.keys())):
                    if idx < len(indices):
                        g_idx = indices[idx]
                        qs = [int(x) for x in res[key] if str(x).isdigit()]
                        if qs:
                            FULL_MAP[g_idx] = qs
                            job_log(job_id, f"   -> Found Qs: {qs}")
            
            time.sleep(1.0)
            
        # PHASE 2: EXTRACT & JSON
        job_log(job_id, "✂️ Cropping & Generating JSON...")
        questions_json = extract_questions(job_id, doc, FULL_MAP)
        
        if not questions_json:
            raise Exception("No questions found after scanning.")
            
        # --- FIXED DATA.JSON STRUCTURE ---
        final_structure = {
            "testConfig": {
                "pdfFileHash": pdf_hash
            },
            "pdfCropperData": {
                "Stark": {
                    "Stark": questions_json
                }
            },
            "appVersion": "1.30.0",
            "generatedBy": "Team_Stark_V33_Groq"
        }
        
        with open(os.path.join(EXPORT_DIR, "data.json"), "w") as f:
            json.dump(final_structure, f, indent=2)
            
        # PHASE 3: ZIP
        job_log(job_id, "📦 Zipping Artifacts...")
        zip_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_result.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(EXPORT_DIR):
                for file in files:
                    z.write(os.path.join(root, file), file)
                    
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["file"] = zip_path
        job_log(job_id, "✅ JOB COMPLETED.")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        job_log(job_id, f"🔥 FATAL: {str(e)}")

# --- 🌐 ROUTES ---
@app.route('/upload', methods=['POST'])
def start():
    if not is_authorized(request): return jsonify({"error": "Auth Failed"}), 403
    if 'pdf' not in request.files: return jsonify({"error": "No PDF"}), 400
    
    f = request.files['pdf']
    jid = str(uuid.uuid4())[:8]
    path = os.path.join(UPLOAD_FOLDER, f"{jid}.pdf")
    f.save(path)
    
    jobs[jid] = {"status": "processing", "logs": [], "file": None}
    threading.Thread(target=process_thread, args=(jid, path)).start()
    
    return jsonify({"job_id": jid})

@app.route('/status/<jid>', methods=['GET'])
def status(jid):
    if jid not in jobs: return jsonify({"error": "404"}), 404
    return jsonify(jobs[jid])

@app.route('/download/<jid>', methods=['GET'])
def download(jid):
    if jid not in jobs or jobs[jid]["status"] != "completed": return jsonify({"error": "Wait"}), 400
    return send_file(jobs[jid]["file"], as_attachment=True, download_name="Stark_Result.zip")

@app.route('/')
def index(): return "Stark V33 (Fixed JSON + Aggressive Groq)"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
