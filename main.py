import os
import sys
import time
import json
import uuid
import shutil
import base64
import threading
import logging
import hashlib
import random
import re
import gc
import zipfile
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from PIL import Image
from groq import Groq, RateLimitError, BadRequestError

# --- CONFIGURATION ---
app = Flask(__name__)
# Allow CORS for all domains as requested
CORS(app, resources={r"/*": {"origins": "*"}}, 
     allow_headers=["*"], 
     expose_headers=["Content-Disposition", "Content-Type"])

# Directories
BASE_TEMP_DIR = "/tmp/stark_processor"
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL STATE ---
jobs = {}
STARK_SECRET = os.environ.get("STARK_SECRET_KEY", "open_access_mode")

# --- MODEL CONFIGURATION ---
# Updated to the specific model requested by user
CURRENT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- SMARTER GROQ MANAGER ---
class SmartGroqManager:
    def __init__(self):
        raw_keys = os.environ.get("GROQ_API_KEYS", "")
        self.all_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        # Dictionary to track when a key will be ready again (timestamp)
        self.key_cooldowns = {k: 0 for k in self.all_keys}
        
        if not self.all_keys:
            logger.warning("⚠️ No GROQ_API_KEYS found in environment variables.")

    def get_client(self):
        if not self.all_keys:
            return None, None
        
        now = time.time()
        # Find keys that are not in cooldown
        available_keys = [k for k in self.all_keys if now >= self.key_cooldowns[k]]
        
        if not available_keys:
            # If all are in cooldown, pick the one that expires soonest
            selected_key = min(self.key_cooldowns, key=self.key_cooldowns.get)
            wait_time = max(0, self.key_cooldowns[selected_key] - now)
            if wait_time > 0:
                logger.info(f"⏳ All keys busy. Waiting {wait_time:.1f}s for key release...")
                time.sleep(wait_time)
        else:
            selected_key = random.choice(available_keys)
            
        return Groq(api_key=selected_key), selected_key

    def mark_rate_limited(self, key):
        """Put a key in penalty box for 20 seconds if it hits rate limit"""
        self.key_cooldowns[key] = time.time() + 20
        logger.warning(f"⚠️ Key ending in ...{key[-4:]} hit rate limit. Cooldown 20s.")

    def call_vision_batch(self, message_payload, retries=5):
        for attempt in range(retries):
            client, key_used = self.get_client()
            if not client: return None

            try:
                completion = client.chat.completions.create(
                    model=CURRENT_VISION_MODEL,
                    messages=[{"role": "user", "content": message_payload}],
                    temperature=0.1,
                    max_tokens=4096,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"}
                )
                return json.loads(completion.choices[0].message.content)

            except RateLimitError:
                self.mark_rate_limited(key_used)
                time.sleep(1) # Short sleep before retry with new key
            except BadRequestError as e:
                # Handle model decommissioning specifically
                if "decommissioned" in str(e).lower() or "model_decommissioned" in str(e).lower():
                     logger.error(f"🔥 FATAL MODEL ERROR: {CURRENT_VISION_MODEL} is decommissioned. Update CURRENT_VISION_MODEL in app.py")
                     # Break loop as retrying won't fix a decommissioned model
                     break 
                logger.error(f"❌ Bad Request: {str(e)}")
                # If image is too large (413), we might want to catch that, but generic logging covers it.
                break
            except Exception as e:
                logger.error(f"❌ Groq Error: {str(e)}")
                time.sleep(1)
        
        return None

groq_manager = SmartGroqManager()

# --- UTILS ---

def is_authorized(req):
    if STARK_SECRET == "open_access_mode": return True
    return req.headers.get("x-stark-secret") == STARK_SECRET

def get_pdf_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

def pixmap_to_base64(pix):
    """Convert PyMuPDF Pixmap to base64 string in memory"""
    try:
        # Optimization: Use JPEG instead of PNG.
        # The new models have a strict 4MB base64 limit. 
        # PNGs often exceed this. JPEGs are much safer.
        if pix.alpha:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        
        img_data = pix.tobytes("jpeg", jpg_quality=85)
        return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error: {e}")
        return ""

def log_job(job_id, msg):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(f"[{job_id}] {msg}")
    if job_id in jobs:
        jobs[job_id]["logs"].append(entry)

# --- CORE LOGIC ---

def get_questions_ai(job_id, doc, page_indices):
    payload_content = [
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

    valid_map = {} # Maps "img_X" to actual page index
    
    for i, p_idx in enumerate(page_indices):
        try:
            page = doc[p_idx]
            # 100 DPI is sufficient for AI reading and keeps size low
            pix = page.get_pixmap(dpi=100)
            b64 = pixmap_to_base64(pix)
            
            if not b64: continue

            key = f"img_{i}"
            valid_map[key] = p_idx
            
            payload_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        except: pass

    if not valid_map: return None

    response = groq_manager.call_vision_batch(payload_content)
    
    # Map back generic keys "img_0" to actual page indices
    result_map = {}
    if response:
        for img_key, q_list in response.items():
            if img_key in valid_map:
                real_page_idx = valid_map[img_key]
                clean_qs = [int(x) for x in q_list if str(x).isdigit()]
                result_map[real_page_idx] = clean_qs
                
    return result_map

def extract_and_stitch(job_id, doc, page_map, export_dir):
    # Flatten map
    all_qs_targets = []
    for p_idx, qs in page_map.items():
        for q in qs:
            all_qs_targets.append({"page": p_idx, "val": q})
    
    # Sort
    all_qs_targets.sort(key=lambda x: (x["page"], x["val"]))

    # Regexes
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
        
        page = doc[p_idx]
        blocks = page.get_text("blocks")
        found = False
        
        for b in blocks:
            text = b[4].strip()
            if not text: continue
            if any(bad in text for bad in BAD_KEYWORDS): continue
            
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

    valid_qs_coords.sort(key=lambda x: (x["page"], x["y0"]))
    log_job(job_id, f"✅ Located {len(valid_qs_coords)} valid start points.")

    # 2. CROP & STITCH
    final_output = []
    
    for i, q in enumerate(valid_qs_coords):
        curr_p = q["page"]
        y_start = max(0, q["y0"] - 10)
        
        # Determine Cut Point
        if i + 1 < len(valid_qs_coords):
            next_q = valid_qs_coords[i+1]
            if next_q["page"] == curr_p:
                y_end = next_q["y0"] - 15
            else:
                y_end = doc[curr_p].rect.height - 50
        else:
            y_end = doc[curr_p].rect.height - 50

        # Stitching Logic
        pages_to_process = []
        
        # Scenario A: Same Page
        if i + 1 < len(valid_qs_coords) and valid_qs_coords[i+1]["page"] == curr_p:
             pages_to_process.append((curr_p, y_start, y_end))
        
        # Scenario B: Multi Page
        elif i + 1 < len(valid_qs_coords) and valid_qs_coords[i+1]["page"] > curr_p:
            pages_to_process.append((curr_p, y_start, doc[curr_p].rect.height - 40))
            for gap_p in range(curr_p + 1, valid_qs_coords[i+1]["page"]):
                pages_to_process.append((gap_p, 40, doc[gap_p].rect.height - 40))
            
            next_p_idx = valid_qs_coords[i+1]["page"]
            next_q_y = valid_qs_coords[i+1]["y0"] - 15
            pages_to_process.append((next_p_idx, 40, next_q_y))
        
        # Scenario C: Last Q
        else:
            pages_to_process.append((curr_p, y_start, doc[curr_p].rect.height - 50))

        # Render
        images = []
        total_h = 0
        max_w = 0
        
        for p_idx_s, y_s, y_e in pages_to_process:
            if y_e <= y_s: continue
            page = doc[p_idx_s]
            rect = fitz.Rect(0, y_s, page.rect.width, y_e)
            pix = page.get_pixmap(dpi=200, clip=rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            total_h += img.height
            max_w = max(max_w, img.width)
        
        if not images: continue
        
        final_img = Image.new('RGB', (max_w, total_h), (255, 255, 255))
        current_y = 0
        for img in images:
            final_img.paste(img, (0, current_y))
            current_y += img.height
        
        # --- CRITICAL: MAINTAIN EXACT FILE NAMING ---
        img_filename = f"Stark__--__{q['label']}__--__1.png"
        save_path = os.path.join(export_dir, img_filename)
        
        # Handle duplicates in same job
        if os.path.exists(save_path):
             img_filename = f"Stark__--__{q['label']}_{curr_p}__--__1.png"
             save_path = os.path.join(export_dir, img_filename)

        final_img.save(save_path)
        
        final_output.append({
            "label": q['label'],
            "filename": img_filename
        })
        
    return final_output

def worker_process(job_id, pdf_path, job_dir):
    # CRITICAL: Isolate output for concurrency
    export_dir = os.path.join(job_dir, "master_package")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        log_job(job_id, "🚀 Starting Process...")
        doc = fitz.open(pdf_path)
        pdf_hash = get_pdf_hash(pdf_path)
        
        total_pages = len(doc)
        FULL_PAGE_GUIDANCE = {}
        BATCH_SIZE = 2
        
        # PHASE 1: SCAN
        for i in range(0, total_pages, BATCH_SIZE):
            batch_indices = range(i, min(i + BATCH_SIZE, total_pages))
            log_job(job_id, f"🤖 Scanning Pages {[b+1 for b in batch_indices]}...")
            
            batch_data = get_questions_ai(job_id, doc, list(batch_indices))
            
            if batch_data:
                for k, v in batch_data.items():
                    FULL_PAGE_GUIDANCE[k] = v
                    log_job(job_id, f"   -> Page {k+1}: Found Qs {v}")
            
            time.sleep(0.5)

        # PHASE 2: PROCESS
        log_job(job_id, "✂️ Processing & Stitching...")
        final_qs = extract_and_stitch(job_id, doc, FULL_PAGE_GUIDANCE, export_dir)
        
        if not final_qs:
            raise Exception("No questions extracted.")
            
        # PHASE 3: JSON & ZIP
        log_job(job_id, "📦 Generating Data Package...")
        
        # --- CRITICAL: MAINTAIN EXACT JSON STRUCTURE ---
        data_json = {
            "testConfig": {"pdfFileHash": pdf_hash},
            "pdfCropperData": {"Stark": {"Stark": {}}},
            "appVersion": "1.30.0",
            "generatedBy": "Team_Stark_V33_Fixed"
        }
        
        for q in final_qs:
            key = q['label']
            if "_" in q['filename']:
                 try:
                    # Try to parse original logic if complex filename
                    key_part = q['filename'].split("__--__")[1]
                    key = key_part
                 except: pass

            data_json["pdfCropperData"]["Stark"]["Stark"][key] = {
                "que": key,
                "type": "mcq",
                "marks": {"cm": 4, "im": -1}, # Preserved logic
                "answerOptions": "4",         # Preserved logic
                "pdfData": [{                 # Preserved dummy logic
                    "x1": 5, "x2": 995, "y1": 100, "y2": 500, "page": 1
                }]
            }
        
        with open(os.path.join(export_dir, "data.json"), "w") as f:
            json.dump(data_json, f, indent=2)

        zip_path = os.path.join(job_dir, f"{job_id}_result.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["file_path"] = zip_path
        log_job(job_id, "✅ JOB COMPLETE. Downloading...")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        log_job(job_id, f"🔥 FATAL ERROR: {str(e)}")
    finally:
        if 'doc' in locals(): doc.close()
        try:
            shutil.rmtree(export_dir) # Clean temp images
        except: pass
        gc.collect()

# --- API ROUTES ---

@app.route('/upload', methods=['POST'])
def start_job():
    if not is_authorized(request): return jsonify({"error": "Unauthorized"}), 403
    if 'pdf' not in request.files: return jsonify({"error": "No file"}), 400
    
    file = request.files['pdf']
    job_id = str(uuid.uuid4())[:8]
    
    # CRITICAL: Unique directory per job
    job_dir = os.path.join(BASE_TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    save_path = os.path.join(job_dir, "source.pdf")
    file.save(save_path)
    
    jobs[job_id] = {"status": "processing", "logs": [], "file_path": None, "error": None}
    
    thread = threading.Thread(target=worker_process, args=(job_id, save_path, job_dir))
    thread.daemon = True
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
    return send_file(jobs[job_id]["file_path"], as_attachment=True, download_name='Stark_Result.zip')

@app.route('/')
def home():
    return "Team Stark V33 (Multi-User & Smart Switch) 🚀"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
