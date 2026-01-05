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
import time
import threading
import uuid
import itertools
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

# --- ⚡ SMART KEY ROTATION (ROUND ROBIN) ---
RAW_KEYS = os.environ.get("GROQ_API_KEYS", "")
# List banake cycle bana diya (Infinite Loop Iterator)
KEY_LIST = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]
KEY_CYCLE = itertools.cycle(KEY_LIST) if KEY_LIST else None

def get_next_groq_client():
    if not KEY_CYCLE: return None
    api_key = next(KEY_CYCLE)
    # print(f"🔑 Switching to Key ending in ...{api_key[-4:]}") # Debugging
    return Groq(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

import base64 # Missed import fix

def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def job_log(job_id, msg):
    print(f"[{job_id}] {msg}", file=sys.stdout, flush=True)
    if job_id in jobs:
        timestamp = time.strftime("%H:%M:%S")
        jobs[job_id]["logs"].append(f"[{timestamp}] {msg}")

# --- 🤖 ROBUST GROQ PROCESSOR (AUTO-RETRY) ---
def get_questions_batch(job_id, image_paths):
    MODEL_NAME = "meta-llama/llama-3.2-90b-vision-preview" # Better vision model if available, else standard
    # Fallback to standard if 90b vision not available on your keys: "llama-3.2-11b-vision-preview"

    message_content = [
        {
            "type": "text",
            "text": """
            Analyze these exam pages. Identify Question Numbers.
            OUTPUT ONLY RAW JSON. NO MARKDOWN. NO EXPLANATION.
            
            RULES:
            1. Report EVERY question number (e.g. "1", "2", "Q3", "4.").
            2. Ignore numbers inside solutions.
            3. Return JSON format: { "img_0": [1, 2], "img_1": [3, 4] }
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

    # RETRY LOGIC (Max 5 attempts with different keys)
    max_retries = 5
    for attempt in range(max_retries):
        client = get_next_groq_client()
        if not client: return None

        try:
            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview", # Using 11b for speed/rate limits
                messages=[{"role": "user", "content": message_content}],
                temperature=0.1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                response_format={"type": "json_object"}
            )
            
            content = completion.choices[0].message.content
            
            # 🛡️ JSON CLEANER (Agar Groq ne kuch extra text likha)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
                return json.loads(clean_json)
            else:
                return json.loads(content) # Try direct load

        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg:
                job_log(job_id, f"⚠️ Rate Limit on Key. Switching... (Attempt {attempt+1}/{max_retries})")
                time.sleep(1) # Chota break
                continue # Loop wapis chalega next key ke saath
            else:
                job_log(job_id, f"❌ Groq Error: {str(e)}")
                return None
    
    job_log(job_id, "❌ All API Keys exhausted rate limits.")
    return None

# --- ✂️ EXTRACTOR & STITCHER (CRASH PROOF) ---
def extract_and_stitch(job_id, doc, page_map):
    all_qs_targets = []
    for p, qs in page_map.items():
        for q in qs:
            all_qs_targets.append({"page": p, "val": q})
    
    # Expanded Regex to catch loose formats
    regex_list = [
        r"^\s*(?:Q|Question|Que|No)[\.\s\-]?\s*(\d+)", 
        r"^\s*(\d+)[\.\)\-\:]", 
        r"^\s*(\d+)\s*$"
    ]
    BAD_KEYWORDS = ["Solution", "Detailed Solution", "Correct Answer", "Explanation", "Ans."]

    valid_qs_coords = []
    
    # 1. FIND COORDINATES
    for item in all_qs_targets:
        p_idx = item['page']
        q_target = item['val']
        
        try:
            page = doc[p_idx]
            blocks = page.get_text("blocks")
        except: continue

        found = False
        
        for b in blocks:
            text = b[4].strip()
            if not text: continue
            
            # Anti-Solution Shield
            if any(bad in text for bad in BAD_KEYWORDS): continue
            
            # Check for Number Match
            for pat in regex_list:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        if val == q_target:
                            valid_qs_coords.append({
                                "label": str(val),
                                "page": p_idx,
                                "y0": b[1], 
                                "y1": b[3]
                            })
                            found = True
                            break
                    except: continue
            if found: break
    
    # 🛡️ FATAL CHECK: Agar Regex fail hua toh khali list mat bhejo
    if not valid_qs_coords:
        job_log(job_id, "⚠️ Warning: Text extraction failed. Checking fallback...")
        # (Future: Yaha geometric fallback aa sakta hai, abhi ke liye skip)
        return []

    valid_qs_coords.sort(key=lambda x: (x["page"], x["y0"]))
    job_log(job_id, f"✅ Text Match Success: {len(valid_qs_coords)} questions identified.")

    # 2. CROP & STITCH
    final_output = []
    
    for i, q in enumerate(valid_qs_coords):
        try:
            curr_p = q["page"]
            y_start = max(0, q["y0"] - 10) 
            
            if i + 1 < len(valid_qs_coords):
                next_q = valid_qs_coords[i+1]
                next_p = next_q["page"]
                if next_p == curr_p:
                    y_end = next_q["y0"] - 15
                else:
                    y_end = doc[curr_p].rect.height - 50
            else:
                y_end = doc[curr_p].rect.height - 50

            # Simplistic stitching for robustness
            pages_to_process = []
            if i + 1 < len(valid_qs_coords) and valid_qs_coords[i+1]["page"] > curr_p:
                # Multi-page Logic
                pages_to_process.append((curr_p, y_start, doc[curr_p].rect.height - 40))
                for gap_p in range(curr_p + 1, valid_qs_coords[i+1]["page"]):
                    pages_to_process.append((gap_p, 40, doc[gap_p].rect.height - 40))
                next_p_idx = valid_qs_coords[i+1]["page"]
                next_q_y = valid_qs_coords[i+1]["y0"] - 15
                pages_to_process.append((next_p_idx, 40, next_q_y))
            else:
                # Single page
                pages_to_process.append((curr_p, y_start, y_end))

            images = []
            total_h = 0
            max_w = 0
            
            for p_idx, y_s, y_e in pages_to_process:
                if y_e <= y_s: continue
                page = doc[p_idx]
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                scale_h = pix.height / page.rect.height
                scale_w = pix.width / page.rect.width
                
                crop_box = (0, y_s * scale_h, pix.width, y_e * scale_h)
                cropped = img.crop(crop_box)
                images.append(cropped)
                total_h += cropped.height
                max_w = max(max_w, cropped.width)
            
            if not images: continue
            
            final_img = Image.new('RGB', (max_w, total_h), (255, 255, 255))
            current_y = 0
            for img in images:
                final_img.paste(img, (0, current_y))
                current_y += img.height
            
            img_filename = f"Stark__--__{q['label']}__--__1.png"
            save_path = os.path.join(EXPORT_DIR, img_filename)
            
            # Anti-Overwrite
            if os.path.exists(save_path):
                img_filename = f"Stark__--__{q['label']}_dup_{uuid.uuid4().hex[:4]}__--__1.png"
                save_path = os.path.join(EXPORT_DIR, img_filename)

            final_img.save(save_path)
            
            final_output.append({
                "label": q['label'],
                "filename": img_filename
            })
        except Exception as e:
            print(f"Skipping Q{q['label']} due to error: {e}")
            continue # Skip bad question, don't crash job
            
    return final_output

# --- 🧵 BACKGROUND WORKER ---
def process_pdf_thread(job_id, pdf_path):
    try:
        job_log(job_id, "🚀 Starting Smart Process...")
        
        if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
        os.makedirs(EXPORT_DIR, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        pdf_hash = get_pdf_hash(pdf_path)
        
        total_pages = len(doc)
        FULL_PAGE_GUIDANCE = {}
        BATCH_SIZE = 2 # Small batch to save memory/tokens
        
        # PHASE 1: SCAN
        for i in range(0, total_pages, BATCH_SIZE):
            batch_indices = range(i, min(i + BATCH_SIZE, total_pages))
            img_paths = []
            
            job_log(job_id, f"🤖 Scanning Pages {[b+1 for b in batch_indices]}...")
            for p_idx in batch_indices:
                pix = doc[p_idx].get_pixmap(dpi=100) # Low DPI for AI (faster)
                path = os.path.join(UPLOAD_FOLDER, f"{job_id}_batch_{p_idx}.jpg")
                pix.save(path)
                img_paths.append(path)
            
            batch_data = get_questions_batch(job_id, img_paths)
            
            if batch_data:
                for rel_idx, key in enumerate(sorted(batch_data.keys())):
                    if rel_idx < len(batch_indices):
                        g_idx = batch_indices[rel_idx]
                        raw_qs = batch_data[key]
                        # Clean non-integers
                        clean_qs = []
                        for x in raw_qs:
                            if isinstance(x, int): clean_qs.append(x)
                            elif isinstance(x, str) and x.isdigit(): clean_qs.append(int(x))
                        
                        if clean_qs:
                            FULL_PAGE_GUIDANCE[g_idx] = clean_qs
                            job_log(job_id, f"   -> Found Qs: {clean_qs}")
            
            # No forced sleep needed with smart rotation, but keep small buffer
            time.sleep(0.5) 
        
        # PHASE 2: PROCESS
        job_log(job_id, "✂️ Stitching Images...")
        final_qs = extract_and_stitch(job_id, doc, FULL_PAGE_GUIDANCE)
        
        if not final_qs:
             # Graceful exit instead of crash
             job_log(job_id, "❌ No questions could be stitched. Try a clearer PDF.")
             jobs[job_id]["status"] = "failed"
             jobs[job_id]["error"] = "Text extraction failed. PDF might be an image scan."
             return
            
        # PHASE 3: JSON & ZIP
        job_log(job_id, "📦 Packing ZIP...")
        
        data_json = {
            "testConfig": {"pdfFileHash": pdf_hash},
            "pdfCropperData": {"Stark": {"Stark": {}}},
            "appVersion": "1.30.0",
            "generatedBy": "Team_Stark_V33_Robust"
        }
        
        for q in final_qs:
            key = q['label'] 
            # Safe Filename Split
            try:
                if "__--__" in q['filename']:
                    key = q['filename'].split("__--__")[1]
            except: pass

            data_json["pdfCropperData"]["Stark"]["Stark"][key] = {
                "que": key, 
                "type": "mcq", 
                "marks": {"cm": 4, "im": -1},
                "answerOptions": "4",
                "pdfData": [{"x1": 5, "x2": 995, "y1": 100, "y2": 500, "page": 1}]
            }
        
        with open(os.path.join(EXPORT_DIR, "data.json"), "w") as f:
            json.dump(data_json, f, indent=2)

        zip_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_result.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(EXPORT_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file), file)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["file"] = zip_path
        job_log(job_id, "✅ JOB COMPLETE.")
        
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
    return "Team Stark V33 (Auto-Retry + Round Robin) 🚀"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
