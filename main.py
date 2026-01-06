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

# --- SYSTEM LIMITS ---
MAX_CONCURRENT_JOBS = 3
MAX_PAGES_PER_PDF = 20

# Semaphore to control active jobs (Queue System)
job_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)

# --- LOGGING ---
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
            selected_key = random.choice(self.all_keys)
        else:
            selected_key = random.choice(available_keys)
            
        return Groq(api_key=selected_key), selected_key

    def mark_rate_limited(self, key):
        self.key_cooldowns[key] = time.time() + 5
        print(f"[SYSTEM] ⚠️ Rate limit on key ...{key[-4:]}. Pausing it for 5s.", flush=True)

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
    UNIVERSAL COORDINATE DETECTION (SEQUENTIAL MODE)
    """
    payload_content = [
        {
            "type": "text",
            "text": """
            Analyze these exam pages.
            
            TASK:
            Identify the FULL BOUNDING BOX for every question block found.
            
            UNIVERSAL LAYOUT INSTRUCTIONS:
            1. **BE GENEROUS**: It is better to include extra whitespace than to cut off text.
            2. **Start** at the Question Number.
            3. **End** below the last option (A, B, C, D) or Answer block.
            4. **WIDTH**:
               - If the text looks like it spans the whole page, set x_start near 0 and x_end near 1000.
               - If it is in a column, capture the FULL column width.
            
            OUTPUT RULES:
            Return JSON. Key: "img_0", Value: LIST of objects:
            {
               "q_num": Integer,
               "x_start": Integer (0-1000),
               "y_start": Integer (0-1000),
               "x_end": Integer (0-1000),
               "y_end": Integer (0-1000)
            }
            
            MULTI-PAGE RULE:
            - If a question continues to the next page, set "y_end" to 1000.
            """
        }
    ]

    valid_map = {}
    
    for i, p_idx in enumerate(page_indices):
        try:
            page = doc[p_idx]
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
    
    result_map = {}
    if response:
        for img_key, item_list in response.items():
            if img_key in valid_map:
                real_page_idx = valid_map[img_key]
                cleaned_items = []
                
                if isinstance(item_list, list):
                    for item in item_list:
                        try:
                            q_num = int(str(item.get("q_num", "")).strip())
                            x0 = int(str(item.get("x_start", "0")).strip())
                            y0 = int(str(item.get("y_start", "0")).strip())
                            x1 = int(str(item.get("x_end", "1000")).strip())
                            y1 = int(str(item.get("y_end", "1000")).strip())
                            
                            # Safety Fixes for coordinate inversion
                            if x1 <= x0: x1 = 1000
                            if y1 <= y0: y1 = y0 + 100
                            
                            cleaned_items.append({
                                "q": q_num, 
                                "x0": x0, "y0": y0, 
                                "x1": x1, "y1": y1
                            })
                        except: pass
                
                cleaned_items.sort(key=lambda x: (x["y0"], x["x0"]))
                result_map[real_page_idx] = cleaned_items
                
    return result_map

def extract_and_stitch_pure_vision(job_id, doc, vision_map, export_dir):
    all_qs_coords = []
    
    for p_idx, items in vision_map.items():
        page = doc[p_idx]
        w = page.rect.width
        h = page.rect.height
        
        for item in items:
            # UNIVERSAL SNAP-TO-EDGE LOGIC
            # If AI says we are close to the edge, Snap to the edge.
            # This fixes "Single Column" cutoffs perfectly.
            
            # Snap X-Start
            if item['x0'] < 100: 
                item['x0'] = 0
            
            # Snap X-End
            if item['x1'] > 900: 
                item['x1'] = 1000
            
            # Convert 0-1000 scale to actual PDF points
            x0 = (item['x0'] / 1000.0) * w
            y0 = (item['y0'] / 1000.0) * h
            x1 = (item['x1'] / 1000.0) * w
            y1 = (item['y1'] / 1000.0) * h
            
            all_qs_coords.append({
                "label": str(item['q']),
                "page": p_idx,
                "rect": fitz.Rect(x0, y0, x1, y1),
                "raw_y1_score": item['y1']
            })

    all_qs_coords.sort(key=lambda x: (x["page"], x["rect"].y0))
    log_job(job_id, f"✅ AI identified {len(all_qs_coords)} bounding boxes.", "INFO")
    
    if not all_qs_coords: return []

    final_output = []
    
    # GENERALLY INCREASED PADDING TO BE SAFE
    PAD_Y_TOP = 40  
    PAD_Y_BOTTOM = 50 
    PAD_X = 30  
    
    for i, q in enumerate(all_qs_coords):
        curr_p = q["page"]
        orig_rect = q["rect"]
        
        safe_x0 = max(0, orig_rect.x0 - PAD_X)
        safe_y0 = max(0, orig_rect.y0 - PAD_Y_TOP)
        safe_x1 = min(doc[curr_p].rect.width, orig_rect.x1 + PAD_X)
        safe_y1 = min(doc[curr_p].rect.height, orig_rect.y1 + PAD_Y_BOTTOM)
        
        is_multipage = False
        
        if q["raw_y1_score"] >= 980:
             is_multipage = True
             safe_y1 = doc[curr_p].rect.height - 40
        elif i + 1 < len(all_qs_coords):
             next_q = all_qs_coords[i+1]
             if next_q["page"] > curr_p and q["raw_y1_score"] > 900:
                 is_multipage = True
                 safe_y1 = doc[curr_p].rect.height - 40

        segments = []
        
        # Segment 1
        segments.append({
            "page": curr_p,
            "rect": fitz.Rect(safe_x0, safe_y0, safe_x1, safe_y1)
        })
        
        # Segment 2+ (Stitching)
        if is_multipage and i + 1 < len(all_qs_coords):
            next_q = all_qs_coords[i+1]
            next_q_page = next_q["page"]
            
            for gap_p in range(curr_p + 1, next_q_page):
                segments.append({
                    "page": gap_p,
                    "rect": fitz.Rect(safe_x0, 40, safe_x1, doc[gap_p].rect.height - 40)
                })
            
            next_q_y = next_q["rect"].y0 - 20
            next_q_y = max(50, next_q_y)
            segments.append({
                "page": next_q_page,
                "rect": fitz.Rect(safe_x0, 40, safe_x1, next_q_y)
            })

        images = []
        total_h = 0
        max_w = 0
        
        for seg in segments:
            page = doc[seg['page']]
            pix = page.get_pixmap(dpi=200, clip=seg['rect'])
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
        
        img_filename = f"Stark__--__{q['label']}__--__1.png"
        save_path = os.path.join(export_dir, img_filename)
        
        if os.path.exists(save_path):
             img_filename = f"Stark__--__{q['label']}_{curr_p}__--__1.png"
             save_path = os.path.join(export_dir, img_filename)

        final_img.save(save_path)
        final_output.append({"label": q['label'], "filename": img_filename})
        
    return final_output

def worker_process(job_id, pdf_path, job_dir):
    # CRITICAL: Acquire Semaphore (Wait in Queue if busy)
    with job_semaphore:
        export_dir = os.path.join(job_dir, "master_package")
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            temp_doc = fitz.open(pdf_path)
            total_pages = len(temp_doc)
            if total_pages > MAX_PAGES_PER_PDF:
                raise Exception(f"PDF too large! Max {MAX_PAGES_PER_PDF} pages allowed.")
            pdf_hash = get_pdf_hash(pdf_path)
            temp_doc.close()
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            return

        try:
            log_job(job_id, f"🚀 STARTING JOB (Sequential). Pages: {total_pages}", "INFO")
            
            FULL_VISION_DATA = {}
            BATCH_SIZE = 4 
            batches = []
            for i in range(0, total_pages, BATCH_SIZE):
                indices = list(range(i, min(i + BATCH_SIZE, total_pages)))
                batches.append(indices)
                
            log_job(job_id, f"⚡ Processing {len(batches)} batches sequentially...", "INFO")

            # --- SEQUENTIAL PROCESSING (NO THREADING) ---
            # This is simpler and less prone to "race conditions" or "silent failures"
            
            for indices in batches:
                log_job(job_id, f"   -> Scanning Pages {[p+1 for p in indices]}...", "INFO")
                
                # Open a FRESH doc handle for every batch to be perfectly safe
                batch_doc = fitz.open(pdf_path)
                try:
                    result = get_questions_ai_coordinates(job_id, batch_doc, indices)
                    if result:
                        for k, v in result.items():
                            FULL_VISION_DATA[k] = v
                            q_nums = [x['q'] for x in v]
                            log_job(job_id, f"      Found Qs: {q_nums}", "INFO")
                    else:
                        log_job(job_id, "      No questions found in this batch.", "WARN")
                    
                    # Polite sleep to respect Groq
                    time.sleep(1) 
                    
                except Exception as e:
                    log_job(job_id, f"      Batch Error: {str(e)}", "ERROR")
                finally:
                    batch_doc.close()

            if not FULL_VISION_DATA:
                 raise Exception("Scan completed but no questions were found in any batch.")

            log_job(job_id, "✂️ Universal Cropping (Snap-to-Edge)...", "INFO")
            
            main_doc = fitz.open(pdf_path)
            final_qs = extract_and_stitch_pure_vision(job_id, main_doc, FULL_VISION_DATA, export_dir)
            main_doc.close()
            
            if not final_qs:
                raise Exception("No valid questions generated after cropping.")
                
            log_job(job_id, "📦 Generating Data Package...", "INFO")
            
            data_json = {
                "testConfig": {"pdfFileHash": pdf_hash},
                "pdfCropperData": {"Stark": {"Stark": {}}},
                "appVersion": "1.30.0",
                "generatedBy": "Team_Stark_Universal_V5"
            }
            
            for q in final_qs:
                key = q['label']
                if "_" in q['filename']:
                     try:
                        key = q['filename'].split("__--__")[1]
                     except: pass

                data_json["pdfCropperData"]["Stark"]["Stark"][key] = {
                    "que": key,
                    "type": "mcq",
                    "marks": {"cm": 4, "im": -1},
                    "answerOptions": "4",
                    "pdfData": [{"x1": 5, "x2": 995, "y1": 100, "y2": 500, "page": 1}]
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
    
    jobs[job_id] = {"status": "queued", "logs": [], "file_path": None, "error": None}
    
    # Spawn thread (It will wait at the Semaphore if busy)
    thread = threading.Thread(target=worker_process, args=(job_id, save_path, job_dir))
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id, "message": "Job queued/started"})

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
    return "Team Stark V33 (Sequential + Snap-to-Edge Universal) 🚀"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
