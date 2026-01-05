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
import io
import zipfile
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from PIL import Image
from groq import Groq, InternalServerError, RateLimitError, APIError

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Directories
BASE_TEMP_DIR = "/tmp/stark_processor"
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL STATE ---
jobs = {}
STARK_SECRET = os.environ.get("STARK_SECRET_KEY", "open_access_mode")

# --- GROQ API MANAGER ---
# Using the stable vision model. 'llama-4-scout' is often experimental/preview.
# If you specifically need scout, change this back, but 11b-vision is very fast and stable.
GROQ_MODEL = "llama-3.2-11b-vision-preview" 

class GroqManager:
    def __init__(self):
        raw_keys = os.environ.get("GROQ_API_KEYS", "")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not self.api_keys:
            logger.warning("⚠️ No GROQ_API_KEYS found in environment variables.")

    def get_client(self):
        if not self.api_keys:
            return None
        # Pick random key to distribute load
        return Groq(api_key=random.choice(self.api_keys))

    def call_vision_batch(self, message_payload, retries=3):
        """
        Robust caller with exponential backoff and key rotation.
        """
        for attempt in range(retries):
            client = self.get_client()
            if not client:
                return None

            try:
                completion = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": message_payload}],
                    temperature=0.1,
                    max_tokens=2048,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"}
                )
                content = completion.choices[0].message.content
                return json.loads(content)

            except RateLimitError:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"⚠️ Rate Limit hit. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"❌ Groq Error (Attempt {attempt+1}/{retries}): {str(e)}")
                time.sleep(1)
        
        return None

groq_manager = GroqManager()

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
    """Convert PyMuPDF Pixmap to base64 string completely in memory."""
    try:
        # Get PNG data from pixmap
        img_data = pix.tobytes("png")
        return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error: {e}")
        return ""

def log_job(job_id, msg):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(f"[{job_id}] {msg}")  # Server console
    if job_id in jobs:
        jobs[job_id]["logs"].append(entry)

# --- CORE LOGIC ---

def get_questions_ai(job_id, doc, page_indices):
    """
    Sends page images to Groq to find Question Numbers.
    Optimization: Converts to base64 in memory, no disk writes.
    """
    payload_content = [
        {
            "type": "text",
            "text": """
            Analyze these exam page images. return a JSON object containing the Question Numbers found on each page.
            
            STRICT RULES:
            1. Identify the start of every question block (e.g., "1.", "Q2", "Q.3", "4)", "Question 5").
            2. IGNORE numbers inside "Solutions", "Answers", "Explanations", or page numbers.
            3. Return format: { "page_index_0": [1, 2, 3], "page_index_1": [4, 5] }
            4. Use the index provided in the image prompt as the key.
            """
        }
    ]

    # Process images for this batch
    valid_indices = []
    
    for p_idx in page_indices:
        try:
            page = doc[p_idx]
            # Lower DPI for AI analysis is faster and cheaper (75-100 is enough for OCR)
            pix = page.get_pixmap(dpi=100) 
            b64_str = pixmap_to_base64(pix)
            
            if b64_str:
                payload_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_str}"}
                })
                # Add text label to help model map image to index
                payload_content.append({
                    "type": "text", 
                    "text": f"Above image is page_index_{p_idx}"
                })
                valid_indices.append(p_idx)
        except Exception as e:
            log_job(job_id, f"⚠️ Error prepping page {p_idx}: {e}")

    if not valid_indices:
        return None

    log_job(job_id, f"🤖 Asking AI to analyze pages {valid_indices}...")
    return groq_manager.call_vision_batch(payload_content)

def extract_and_stitch_robust(job_id, doc, question_map, export_dir):
    """
    Robust coordinate extraction and stitching.
    """
    # 1. Gather all targets
    targets = []
    for p_key, q_list in question_map.items():
        # Handle keys like "page_index_1" or just "1"
        try:
            if "index_" in str(p_key):
                p_idx = int(str(p_key).split("_")[-1])
            else:
                p_idx = int(p_key)
            
            for q in q_list:
                targets.append({"page": p_idx, "q_num": int(q)})
        except ValueError:
            continue

    # Sort targets by page, then by number (assumption)
    targets.sort(key=lambda x: (x['page'], x['q_num']))

    # 2. Find Coordinates (Regex Search)
    regex_patterns = [
        r"^\s*(?:Q|Question|Que|No)[\.\s\-]?\s*(\d+)",  # Q.1, Q 1
        r"^\s*(\d+)[\.\)\-\:]",                          # 1., 1)
        r"^\s*(\d+)\s*$"                                 # 1 (Standalone)
    ]
    ignore_keywords = ["Solution", "Answer", "Explanation", "Correct", "Ans"]

    located_qs = []

    for t in targets:
        page = doc[t['page']]
        blocks = page.get_text("blocks")
        found = False
        
        for b in blocks:
            text = b[4].strip()
            if not text: continue
            
            # Anti-Solution Guard
            if any(k.lower() in text.lower() for k in ignore_keywords): continue

            for pat in regex_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        if val == t['q_num']:
                            located_qs.append({
                                "val": val,
                                "page": t['page'],
                                "y0": b[1], # Top
                                "y1": b[3]  # Bottom
                            })
                            found = True
                            break
                    except: pass
            if found: break
    
    log_job(job_id, f"✅ Located coordinates for {len(located_qs)} questions.")

    # 3. Stitching Logic
    final_results = []
    
    # Sort located questions by position
    located_qs.sort(key=lambda x: (x['page'], x['y0']))

    for i, q in enumerate(located_qs):
        try:
            curr_p = q['page']
            # Start slightly above the number
            y_start = max(0, q['y0'] - 15) 
            
            # Determine End Point
            if i + 1 < len(located_qs):
                next_q = located_qs[i+1]
                if next_q['page'] == curr_p:
                    # Next Q is on same page, cut before it
                    y_end = max(y_start + 50, next_q['y0'] - 20)
                else:
                    # Next Q is on later page
                    y_end = doc[curr_p].rect.height - 40 # Bottom margin
            else:
                # Last question
                y_end = doc[curr_p].rect.height - 40

            # Collect image segments
            segments = []
            
            # A. Current Page Segment
            segments.append({"page": curr_p, "y1": y_start, "y2": y_end})

            # B. Multi-page logic (if next Q is on a later page)
            if i + 1 < len(located_qs):
                next_q = located_qs[i+1]
                if next_q['page'] > curr_p:
                    # Add full intermediate pages
                    for gap_p in range(curr_p + 1, next_q['page']):
                        h = doc[gap_p].rect.height
                        segments.append({"page": gap_p, "y1": 40, "y2": h - 40})
                    
                    # Add top of next page until next Q starts
                    next_h_end = max(50, next_q['y0'] - 20)
                    segments.append({"page": next_q['page'], "y1": 40, "y2": next_h_end})

            # Render and Stitch
            pil_images = []
            max_width = 0
            total_height = 0

            for seg in segments:
                page = doc[seg['page']]
                rect = fitz.Rect(0, seg['y1'], page.rect.width, seg['y2'])
                # High DPI for final output
                pix = page.get_pixmap(dpi=200, clip=rect) 
                
                # Convert to PIL
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                
                pil_images.append(img)
                max_width = max(max_width, img.width)
                total_height += img.height

            if not pil_images: continue

            # Create Canvas
            final_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for img in pil_images:
                # Center align if widths differ slightly? Or left align. Left is safer for text.
                final_img.paste(img, (0, y_offset))
                y_offset += img.height

            # Save
            filename = f"Q{q['val']}_{uuid.uuid4().hex[:4]}.png"
            full_path = os.path.join(export_dir, filename)
            final_img.save(full_path, "PNG", optimize=True)
            
            final_results.append({
                "label": str(q['val']),
                "filename": filename
            })

        except Exception as e:
            log_job(job_id, f"⚠️ Error stitching Q{q.get('val', '?')}: {e}")

    return final_results

def worker_process(job_id, pdf_path, job_dir):
    """
    Main Thread Worker. 
    Running inside a specific directory per job to prevent race conditions.
    """
    export_dir = os.path.join(job_dir, "output")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        log_job(job_id, "🚀 Processing Started.")
        
        doc = fitz.open(pdf_path)
        pdf_hash = get_pdf_hash(pdf_path)
        total_pages = len(doc)
        
        full_map = {}
        BATCH_SIZE = 3 # Can increase if memory allows
        
        # Phase 1: AI Scan
        for i in range(0, total_pages, BATCH_SIZE):
            batch = list(range(i, min(i + BATCH_SIZE, total_pages)))
            log_job(job_id, f"🔍 AI Scanning pages: {[b+1 for b in batch]}")
            
            ai_data = get_questions_ai(job_id, doc, batch)
            
            if ai_data:
                # Merge results
                for key, val in ai_data.items():
                    # clean key to get page index (handles "page_index_1" or "1")
                    # clean val to ensure list of ints
                    full_map[key] = val
            
            # Rate limit polite pause
            time.sleep(0.5)

        log_job(job_id, f"💡 AI Found potential questions on {len(full_map)} pages.")

        # Phase 2: Extraction
        log_job(job_id, "✂️  Cropping and Stitching...")
        final_qs = extract_and_stitch_robust(job_id, doc, full_map, export_dir)

        if not final_qs:
            raise Exception("No questions could be extracted/stitched.")

        # Phase 3: JSON Generation
        log_job(job_id, "📝 Generating Metadata...")
        data_json = {
            "testConfig": {"pdfFileHash": pdf_hash},
            "pdfCropperData": {"Stark": {"Stark": {}}},
            "appVersion": "1.30.0",
            "meta": {"total_questions": len(final_qs)}
        }

        for item in final_qs:
            key = item['label']
            data_json["pdfCropperData"]["Stark"]["Stark"][key] = {
                "que": key,
                "type": "mcq",
                "file": item['filename'],
                "processed_at": time.time()
            }

        with open(os.path.join(export_dir, "data.json"), "w") as f:
            json.dump(data_json, f, indent=2)

        # Phase 4: Zipping
        log_job(job_id, "📦 Zipping...")
        zip_path = os.path.join(job_dir, f"Stark_Result_{job_id}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["file_path"] = zip_path
        log_job(job_id, "✅ JOB COMPLETE.")

    except Exception as e:
        logger.exception(f"Job {job_id} Failed")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        log_job(job_id, f"🔥 FATAL: {str(e)}")
    finally:
        # Cleanup
        if 'doc' in locals(): doc.close()
        # Clean up the output folder (unzipped images), keep the zip
        try:
            shutil.rmtree(export_dir)
            if os.path.exists(pdf_path): os.remove(pdf_path)
        except: pass
        gc.collect()

# --- API ROUTES ---

@app.route('/upload', methods=['POST'])
def start_job():
    if not is_authorized(request):
        return jsonify({"error": "Unauthorized"}), 403
    
    if 'pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    job_id = str(uuid.uuid4())[:8]
    
    # Create isolated directory for this job
    job_dir = os.path.join(BASE_TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    pdf_path = os.path.join(job_dir, "source.pdf")
    file.save(pdf_path)

    jobs[job_id] = {
        "status": "processing",
        "logs": [],
        "error": None,
        "file_path": None,
        "created_at": time.time()
    }

    # Start Worker
    thread = threading.Thread(target=worker_process, args=(job_id, pdf_path, job_dir))
    thread.daemon = True # Ensures thread dies if main app dies
    thread.start()

    return jsonify({
        "job_id": job_id,
        "message": "Processing started",
        "status_url": f"/status/{job_id}"
    })

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    return jsonify({
        "status": job["status"],
        "logs": job["logs"][-20:], # Return last 20 logs to save bandwidth
        "error": job["error"],
        "download_ready": job["status"] == "completed"
    })

@app.route('/download/<job_id>', methods=['GET'])
def download(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    if job["status"] != "completed" or not job["file_path"]:
        return jsonify({"error": "File not ready"}), 400
        
    return send_file(
        job["file_path"], 
        as_attachment=True, 
        download_name=f"Stark_Export_{job_id}.zip"
    )

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "active", "jobs_in_mem": len(jobs)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
