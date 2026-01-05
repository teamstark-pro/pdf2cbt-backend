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
CORS(app, resources={r"/*": {"origins": "*"}}, 
     allow_headers=["*"], 
     expose_headers=["Content-Disposition", "Content-Type"])

# Directories
BASE_TEMP_DIR = "/tmp/stark_processor"
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# --- IMPROVED LOGGING FOR RAILWAY ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StarkLogger")

def log_job(job_id, msg, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{job_id}] [{level}] {msg}", file=sys.stdout, flush=True)
    if job_id in jobs:
        jobs[job_id]["logs"].append(f"[{timestamp}] {msg}")

# --- GLOBAL STATE ---
jobs = {}
STARK_SECRET = os.environ.get("STARK_SECRET_KEY", "open_access_mode")

# --- MODEL CONFIGURATION ---
CURRENT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- SMARTER GROQ MANAGER ---
class SmartGroqManager:
    def __init__(self):
        raw_keys = os.environ.get("GROQ_API_KEYS", "")
        self.all_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.key_cooldowns = {k: 0 for k in self.all_keys}
        if not self.all_keys:
            log_job("SYSTEM", "⚠️ No GROQ_API_KEYS found!", "WARN")

    def get_client(self):
        if not self.all_keys: return None, None
        now = time.time()
        available_keys = [k for k in self.all_keys if now >= self.key_cooldowns[k]]
        if not available_keys:
            selected_key = min(self.key_cooldowns, key=self.key_cooldowns.get)
            wait_time = max(0, self.key_cooldowns[selected_key] - now)
            if wait_time > 0:
                print(f"[SYSTEM] ⏳ Keys busy. Sleeping {wait_time:.1f}s...", flush=True)
                time.sleep(wait_time)
        else:
            selected_key = random.choice(available_keys)
        return Groq(api_key=selected_key), selected_key

    def mark_rate_limited(self, key):
        self.key_cooldowns[key] = time.time() + 20
        print(f"[SYSTEM] ⚠️ Rate limit on key ...{key[-4:]}. Cooldown 20s.", flush=True)

    def call_vision_batch(self, message_payload, retries=5):
        for attempt in range(retries):
            client, key_used = self.get_client()
            if not client: return None
            try:
                completion = client.chat.completions.create(
                    model=CURRENT_VISION_MODEL,
                    messages=[{"role": "user", "content": message_payload}],
                    temperature=0.1, max_tokens=4096, top_p=1, stream=False,
                    response_format={"type": "json_object"}
                )
                return json.loads(completion.choices[0].message.content)
            except RateLimitError:
                self.mark_rate_limited(key_used)
                time.sleep(1)
            except BadRequestError as e:
                if "decommissioned" in str(e).lower():
                     log_job("SYSTEM", f"🔥 FATAL: Model {CURRENT_VISION_MODEL} decommissioned.", "ERROR")
                     break 
                log_job("SYSTEM", f"❌ Bad Request: {str(e)}", "ERROR")
                break
            except Exception as e:
                log_job("SYSTEM", f"❌ Groq Error: {str(e)}", "ERROR")
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
    try:
        if pix.alpha: pix = fitz.Pixmap(fitz.csRGB, pix)
        img_data = pix.tobytes("jpeg", jpg_quality=85)
        return base64.b64encode(img_data).decode('utf-8')
    except Exception as e: return ""

# --- CORE LOGIC ---

def get_questions_ai_coordinates(job_id, doc, page_indices):
    """
    Asks Groq to return not just the question number, but the Y-COORDINATE (0-1000 scale).
    This removes the need for Regex searching later.
    """
    payload_content = [
        {
            "type": "text",
            "text": """
            Analyze these exam pages.
            
            TASK:
            Identify the START of every question.
            
            CRITICAL OUTPUT RULES:
            1. Return a JSON object where keys are the image labels (e.g., "img_0").
            2. The value must be a LIST of objects containing:
               - "q_num": The question number (integer).
               - "y_start": The vertical Y-coordinate where this question starts (on a scale of 0 to 1000). 0 is top, 1000 is bottom.
            
            EXAMPLE OUTPUT:
            {
              "img_0": [
                {"q_num": 1, "y_start": 50},
                {"q_num": 2, "y_start": 450}
              ]
            }
            
            IGNORE:
            - Solutions, Answers, Explanations.
            - Headers/Footers.
            """
        }
    ]

    valid_map = {}
    
    for i, p_idx in enumerate(page_indices):
        try:
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=100) # Low DPI is fine for layout analysis
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
    
    result_map = {}
    if response:
        for img_key, item_list in response.items():
            if img_key in valid_map:
                real_page_idx = valid_map[img_key]
                cleaned_items = []
                
                # Robust parsing of AI response
                if isinstance(item_list, list):
                    for item in item_list:
                        try:
                            # Handle string inputs gracefully
                            q_num = int(str(item.get("q_num", "")).strip())
                            y_start = int(str(item.get("y_start", "0")).strip())
                            cleaned_items.append({"q": q_num, "y": y_start})
                        except: pass
                
                # Sort by Y position to be safe
                cleaned_items.sort(key=lambda x: x["y"])
                result_map[real_page_idx] = cleaned_items
                
    return result_map

def extract_and_stitch_pure_vision(job_id, doc, vision_map, export_dir):
    """
    Uses ONLY the coordinates provided by Groq. No Regex.
    """
    # Flatten map into a sortable list
    all_qs_coords = []
    
    for p_idx, items in vision_map.items():
        page_h = doc[p_idx].rect.height
        
        for item in items:
            # Convert 0-1000 scale to actual PDF points
            normalized_y = item['y']
            actual_y = (normalized_y / 1000.0) * page_h
            
            all_qs_coords.append({
                "label": str(item['q']),
                "page": p_idx,
                "y0": actual_y
            })

    # Sort: Primary by Page, Secondary by Y-position
    all_qs_coords.sort(key=lambda x: (x["page"], x["y0"]))
    
    log_job(job_id, f"✅ AI identified {len(all_qs_coords)} start points directly.", "INFO")
    
    if not all_qs_coords:
        return []

    # 2. CROP & STITCH
    final_output = []
    
    for i, q in enumerate(all_qs_coords):
        curr_p = q["page"]
        
        # Buffer: Start slightly above the AI's detected point to catch the top of the number
        y_start = max(0, q["y0"] - 15)
        
        # Determine End Point (Cut Logic)
        if i + 1 < len(all_qs_coords):
            next_q = all_qs_coords[i+1]
            if next_q["page"] == curr_p:
                # Next question is on the same page
                # Cut slightly before the next question starts
                y_end = max(y_start + 20, next_q["y0"] - 10)
            else:
                # Next question is on a later page
                # Take until bottom of current page
                y_end = doc[curr_p].rect.height - 40 # Bottom margin buffer
        else:
            # Last question of the whole doc
            y_end = doc[curr_p].rect.height - 40

        # Stitching Logic (Handling Multi-Page Questions)
        pages_to_process = []
        
        # Scenario A: Single Page Segment
        if i + 1 < len(all_qs_coords) and all_qs_coords[i+1]["page"] == curr_p:
             pages_to_process.append((curr_p, y_start, y_end))
        
        # Scenario B: Multi-Page Segment
        elif i + 1 < len(all_qs_coords) and all_qs_coords[i+1]["page"] > curr_p:
            # 1. Remainder of Current Page
            pages_to_process.append((curr_p, y_start, doc[curr_p].rect.height - 40))
            
            # 2. Full Intermediate Pages
            for gap_p in range(curr_p + 1, all_qs_coords[i+1]["page"]):
                pages_to_process.append((gap_p, 40, doc[gap_p].rect.height - 40))
            
            # 3. Top of Next Page (until next Q starts)
            next_p_idx = all_qs_coords[i+1]["page"]
            next_q_y = all_qs_coords[i+1]["y0"] - 15
            # Ensure we don't have negative height
            next_q_y = max(50, next_q_y) 
            pages_to_process.append((next_p_idx, 40, next_q_y))
            
        # Scenario C: Very Last Question
        else:
            pages_to_process.append((curr_p, y_start, doc[curr_p].rect.height - 40))

        # Render Images
        images = []
        total_h = 0
        max_w = 0
        
        for p_idx_s, y_s, y_e in pages_to_process:
            if y_e <= y_s: continue # Skip invalid segments
            
            page = doc[p_idx_s]
            rect = fitz.Rect(0, y_s, page.rect.width, y_e)
            
            # High DPI for final output quality
            pix = page.get_pixmap(dpi=200, clip=rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            images.append(img)
            total_h += img.height
            max_w = max(max_w, img.width)
        
        if not images: continue
        
        # Stitch
        final_img = Image.new('RGB', (max_w, total_h), (255, 255, 255))
        current_y = 0
        for img in images:
            final_img.paste(img, (0, current_y))
            current_y += img.height
        
        # Save
        img_filename = f"Stark__--__{q['label']}__--__1.png"
        save_path = os.path.join(export_dir, img_filename)
        
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
    export_dir = os.path.join(job_dir, "master_package")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        log_job(job_id, "🚀 Starting Process...", "INFO")
        doc = fitz.open(pdf_path)
        pdf_hash = get_pdf_hash(pdf_path)
        
        total_pages = len(doc)
        FULL_VISION_DATA = {}
        BATCH_SIZE = 2
        
        # PHASE 1: VISION SCAN (Coordinates)
        for i in range(0, total_pages, BATCH_SIZE):
            batch_indices = range(i, min(i + BATCH_SIZE, total_pages))
            log_job(job_id, f"🤖 Vision Scanning Pages {[b+1 for b in batch_indices]}...", "INFO")
            
            # Now returns coordinates too!
            batch_data = get_questions_ai_coordinates(job_id, doc, list(batch_indices))
            
            if batch_data:
                for k, v in batch_data.items():
                    FULL_VISION_DATA[k] = v
                    # Log concise summary
                    q_nums = [x['q'] for x in v]
                    log_job(job_id, f"   -> Page {k+1}: Found Qs {q_nums}", "INFO")
            
            time.sleep(0.5)

        # PHASE 2: CROP (Pure Vision)
        log_job(job_id, "✂️ Cropping based on Vision Coordinates...", "INFO")
        final_qs = extract_and_stitch_pure_vision(job_id, doc, FULL_VISION_DATA, export_dir)
        
        if not final_qs:
            raise Exception("AI found no valid questions.")
            
        # PHASE 3: JSON & ZIP
        log_job(job_id, "📦 Generating Data Package...", "INFO")
        
        data_json = {
            "testConfig": {"pdfFileHash": pdf_hash},
            "pdfCropperData": {"Stark": {"Stark": {}}},
            "appVersion": "1.30.0",
            "generatedBy": "Team_Stark_Vision_Mode"
        }
        
        for q in final_qs:
            key = q['label']
            if "_" in q['filename']:
                 try:
                    key_part = q['filename'].split("__--__")[1]
                    key = key_part
                 except: pass

            data_json["pdfCropperData"]["Stark"]["Stark"][key] = {
                "que": key,
                "type": "mcq",
                "marks": {"cm": 4, "im": -1},
                "answerOptions": "4",
                "pdfData": [{
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
        log_job(job_id, "✅ JOB COMPLETE. Ready for download.", "SUCCESS")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        log_job(job_id, f"🔥 FATAL ERROR: {str(e)}", "ERROR")
    finally:
        if 'doc' in locals(): doc.close()
        try:
            shutil.rmtree(export_dir)
        except: pass
        gc.collect()

# --- API ROUTES ---

@app.route('/upload', methods=['POST'])
def start_job():
    if not is_authorized(request): return jsonify({"error": "Unauthorized"}), 403
    if 'pdf' not in request.files: return jsonify({"error": "No file"}), 400
    
    file = request.files['pdf']
    job_id = str(uuid.uuid4())[:8]
    
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
    return "Team Stark V33 (Pure Vision Mode) 🚀"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
