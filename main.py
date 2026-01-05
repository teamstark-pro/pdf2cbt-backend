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

# --- 🔥 CORS & SECURITY 🔥 ---
CORS(app, resources={r"/*": {"origins": "*"}}, 
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["*"],
     expose_headers=["Content-Disposition", "Content-Type"])

EXPORT_DIR = "/tmp/cbt_master_package"
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

jobs = {}

# --- 🔐 SECURITY CHECK ---
STARK_SECRET = os.environ.get("STARK_SECRET_KEY", "open_access_mode")
def is_authorized(req):
    if STARK_SECRET == "open_access_mode": return True
    client_key = req.headers.get("x-stark-secret")
    return client_key == STARK_SECRET

# --- ⚡ MULTI-KEY GROQ CLIENT ---
RAW_KEYS = os.environ.get("GROQ_API_KEYS", "")
API_KEY_POOL = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

def get_groq_client():
    if not API_KEY_POOL: return None
    selected_key = random.choice(API_KEY_POOL)
    return Groq(api_key=selected_key)

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

# --- 🤖 GROQ BATCH PROCESSOR ---
def get_questions_batch(job_id, image_paths):
    client = get_groq_client()
    if not client: return None
    
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    try:
        message_content = [
            {
                "type": "text",
                "text": """
                Analyze these exam pages. Identify Question Numbers.
                
                RULES:
                1. Report EVERY question number you see starting a block (e.g., "1.", "Q2", "Q.3", "4)").
                2. IGNORE numbers inside "Solution", "Answer", "Explanation" blocks.
                3. IGNORE page numbers.
                4. Output must be exhaustive.
                
                Return JSON: { "img_0": [1, 2, 3], "img_1": [4, 5] }
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
            temperature=0.1, max_tokens=1000, top_p=1, stream=False,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        return data
    except Exception as e:
        job_log(job_id, f"❌ Groq Error: {str(e)}")
        return None

# --- ✂️ EXTRACTOR & STITCHER ---
def extract_and_stitch(job_id, doc, page_map):
    # Flatten map to a list: [(page, q_num), ...]
    all_qs_targets = []
    for p, qs in page_map.items():
        for q in qs:
            all_qs_targets.append({"page": p, "val": q})
    
    # Regexes (Expanded)
    regex_list = [
        r"^\s*(?:Q|Question|Que|No)[\.\s\-]?\s*(\d+)", # Q.1, Q1
        r"^\s*(\d+)[\.\)\-\:]", # 1., 1)
        r"^\s*(\d+)\s*$" # 1 (Standalone)
    ]
    BAD_KEYWORDS = ["Solution", "Detailed Solution", "Correct Answer", "Explanation", "Ans."]

    valid_qs_coords = []
    
    # 1. FIND COORDINATES
    for item in all_qs_targets:
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
                        # Match Groq's target number
                        if val == q_target:
                            valid_qs_coords.append({
                                "label": str(val),
                                "page": p_idx,
                                "y0": b[1], # Top of block
                                "y1": b[3]  # Bottom of block
                            })
                            found = True
                            break
                    except: continue
            if found: break
    
    # Sort by (Page, Y-Pos)
    valid_qs_coords.sort(key=lambda x: (x["page"], x["y0"]))
    
    job_log(job_id, f"✅ Located {len(valid_qs_coords)} valid start points.")

    # 2. CROP & STITCH
    final_output = []
    
    for i, q in enumerate(valid_qs_coords):
        curr_p = q["page"]
        y_start = max(0, q["y0"] - 10) # Little buffer above
        
        # Determine Cut Point (Next Q or End of Page)
        if i + 1 < len(valid_qs_coords):
            next_q = valid_qs_coords[i+1]
            next_p = next_q["page"]
            
            # If next Q is on same page, cut there
            if next_p == curr_p:
                y_end = next_q["y0"] - 15
            # If next Q is far away, take full page
            else:
                y_end = doc[curr_p].rect.height - 50
        else:
            # Last question
            y_end = doc[curr_p].rect.height - 50

        # STITCHING LOGIC
        pages_to_process = []
        
        # Scenario A: Standard (Single Page)
        if i + 1 < len(valid_qs_coords) and valid_qs_coords[i+1]["page"] == curr_p:
             pages_to_process.append((curr_p, y_start, y_end))
             
        # Scenario B: Split Question (Next Q is on Next Page)
        elif i + 1 < len(valid_qs_coords) and valid_qs_coords[i+1]["page"] > curr_p:
            # Current Page: Start -> Bottom
            pages_to_process.append((curr_p, y_start, doc[curr_p].rect.height - 40))
            
            # Intermediate Pages (Full) - if any
            for gap_p in range(curr_p + 1, valid_qs_coords[i+1]["page"]):
                pages_to_process.append((gap_p, 40, doc[gap_p].rect.height - 40))
                
            # Next Page: Top -> Next Q Start
            next_p_idx = valid_qs_coords[i+1]["page"]
            next_q_y = valid_qs_coords[i+1]["y0"] - 15
            pages_to_process.append((next_p_idx, 40, next_q_y)) # Start from 40 (Header buffer)
            
            job_log(job_id, f"🧵 Stitching Q{q['label']} across segments")
            
        else:
            # Last question on page
            pages_to_process.append((curr_p, y_start, doc[curr_p].rect.height - 50))

        # RENDER IMAGES
        images = []
        total_h = 0
        max_w = 0
        
        for p_idx, y_s, y_e in pages_to_process:
            if y_e <= y_s: continue
            
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            pg_w = page.rect.width
            pg_h = page.rect.height
            scale_w = pix.width / pg_w
            scale_h = pix.height / pg_h
            
            crop_box = (0, y_s * scale_h, pg_w * scale_w, y_e * scale_h)
            try:
                cropped = img.crop(crop_box)
                images.append(cropped)
                total_h += cropped.height
                max_w = max(max_w, cropped.width)
            except: pass
        
        if not images: continue
        
        # Merge
        if len(images) == 1:
            final_img = images[0]
        else:
            final_img = Image.new('RGB', (max_w, total_h), (255, 255, 255))
            current_y = 0
            for img in images:
                final_img.paste(img, (0, current_y))
                current_y += img.height
        
        # Save
        img_filename = f"Stark__--__{q['label']}__--__1.png"
        save_path = os.path.join(EXPORT_DIR, img_filename)
        
        # Overwrite protection
        if os.path.exists(save_path):
             img_filename = f"Stark__--__{q['label']}_{curr_p}__--__1.png"
             save_path = os.path.join(EXPORT_DIR, img_filename)

        final_img.save(save_path)
        
        final_output.append({
            "label": q['label'],
            "filename": img_filename
        })
        
    return final_output

# --- 🧵 BACKGROUND WORKER ---
def process_pdf_thread(job_id, pdf_path):
    try:
        job_log(job_id, "🚀 Starting Process...")
        
        if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        pdf_hash = get_pdf_hash(pdf_path) # Hash is Back!
        
        total_pages = len(doc)
        FULL_PAGE_GUIDANCE = {}
        BATCH_SIZE = 2
        
        # PHASE 1: SCAN
        for i in range(0, total_pages, BATCH_SIZE):
            batch_indices = range(i, min(i + BATCH_SIZE, total_pages))
            img_paths = []
            
            job_log(job_id, f"🤖 Scanning Pages {[b+1 for b in batch_indices]}...")
            for p_idx in batch_indices:
                pix = doc[p_idx].get_pixmap(dpi=100)
                path = os.path.join(UPLOAD_FOLDER, f"{job_id}_batch_{p_idx}.jpg")
                pix.save(path)
                img_paths.append(path)
            
            batch_data = get_questions_batch(job_id, img_paths)
            
            if batch_data:
                for rel_idx, key in enumerate(sorted(batch_data.keys())):
                    if rel_idx < len(batch_indices):
                        g_idx = batch_indices[rel_idx]
                        raw_qs = batch_data[key]
                        clean_qs = [int(x) for x in raw_qs if str(x).isdigit()]
                        if clean_qs:
                            FULL_PAGE_GUIDANCE[g_idx] = clean_qs
                            job_log(job_id, f"   -> Page {g_idx+1}: Found Qs {clean_qs}")
            
            time.sleep(1.0) 
        
        # PHASE 2: PROCESS
        job_log(job_id, "✂️ Processing & Stitching...")
        final_qs = extract_and_stitch(job_id, doc, FULL_PAGE_GUIDANCE)
        
        if not final_qs:
            raise Exception("No questions extracted.")
            
        # PHASE 3: JSON & ZIP
        job_log(job_id, "📦 Generating Data Package...")
        
        # JSON Structure with Hash (Fixed)
        data_json = {
            "testConfig": {"pdfFileHash": pdf_hash},
            "pdfCropperData": {"Stark": {"Stark": {}}},
            "appVersion": "1.30.0",
            "generatedBy": "Team_Stark_V33_Fixed"
        }
        
        for q in final_qs:
            key = q['label'] 
            if "_" in q['filename']:
                 key = q['filename'].split("__--__")[1]

            data_json["pdfCropperData"]["Stark"]["Stark"][key] = {
                "que": key, 
                "type": "mcq", 
                "marks": {"cm": 4, "im": -1},
                "answerOptions": "4",
                # Note: pdfData coordinates are approximate for CBT since we stitched images
                # But we provide dummy valid coords so frontend doesn't crash
                "pdfData": [{
                    "x1": 5, "x2": 995, "y1": 100, "y2": 500, "page": 1
                }]
            }
        
        # 🔥 FIX: WRITING FILE OUTSIDE THE LOOP 🔥
        with open(os.path.join(EXPORT_DIR, "data.json"), "w") as f:
            json.dump(data_json, f, indent=2)

        zip_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_result.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(EXPORT_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file), file)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["file"] = zip_path
        job_log(job_id, "✅ JOB COMPLETE. Downloading...")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        job_log(job_id, f"🔥 FATAL ERROR: {str(e)}")

@app.route('/upload', methods=['POST'])
def start_job():
    if not is_authorized(request): return jsonify({"error": "Unauthorized"}), 403
    if 'pdf' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['pdf']
    job_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.pdf")
    file.save(save_path)
    jobs[job_id] = {"status": "processing", "logs": [], "file": None, "error": None}
    thread = threading.Thread(target=process_pdf_thread, args=(job_id, save_path))
    thread.start()
    return jsonify({"job_id": job_id, "message": "Job started"})

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in jobs: return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": jobs[job_id]["status"],
        "logs": jobs[job_id]["logs"],
        "error": jobs[job_id]["error"]
    })

@app.route('/download/<job_id>', methods=['GET'])
def download_result(job_id):
    if job_id not in jobs or jobs[job_id]["status"] != "completed":
        return jsonify({"error": "Not ready"}), 404
    return send_file(jobs[job_id]["file"], as_attachment=True, download_name='Stark_Result.zip')

@app.route('/')
def home():
    return "Team Stark V33 (Bug Fixed) 🚀"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
