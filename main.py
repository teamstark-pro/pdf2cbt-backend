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

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, 
     allow_headers=["*"], 
     expose_headers=["Content-Disposition", "Content-Type"])

# Directories
BASE_TEMP_DIR = "/tmp/stark_processor"
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# --- SYSTEM LIMITS ---
MAX_CONCURRENT_JOBS = 5
MAX_PAGES_PER_PDF = 50

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

# --- REGEX ENGINE LOGIC ---

def find_questions_via_regex(doc, pdf_type):
    """
    Scans the PDF text layer using Regex.
    Returns a list of Question Blocks with precise coordinates.
    """
    questions = []
    
    # Comprehensive Regex Patterns for Question Numbers
    # 1. "1.", "1)", "Q1", "Q.1", "Question 1"
    patterns = [
        r"^\s*(\d+)[\.\)\-\:]",           # 1. or 1) or 1-
        r"^\s*Q(?:uestion)?[\.\s]*(\d+)", # Q.1 or Question 1
        r"^\s*\((\d+)\)"                  # (1)
    ]
    
    # Keywords to avoid (False Positives)
    bad_keywords = ["Solution", "Answer", "Exp:", "Explanation", "Page", "Fig", "Table"]

    total_pages = len(doc)
    
    for p_idx in range(total_pages):
        page = doc[p_idx]
        width = page.rect.width
        height = page.rect.height
        
        # Get all text blocks: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        
        # --- LAYOUT SORTING STRATEGY ---
        if pdf_type == 'double_col':
            # Split into Left and Right Columns based on midline
            midpoint = width / 2
            left_col = [b for b in blocks if b[0] < midpoint]
            right_col = [b for b in blocks if b[0] >= midpoint]
            
            # Sort each column Top-to-Bottom
            left_col.sort(key=lambda b: b[1])
            right_col.sort(key=lambda b: b[1])
            
            # Merge: Process Left column then Right column
            sorted_blocks = left_col + right_col
        else:
            # Single Column: Just sort Top-to-Bottom
            sorted_blocks = sorted(blocks, key=lambda b: b[1])

        # Scan blocks for Question Starters
        for b in sorted_blocks:
            text = b[4].strip()
            if not text: continue
            
            # Filter bad keywords
            if any(bk in text for bk in bad_keywords): continue
            
            is_match = False
            q_num = -1
            
            for pat in patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    try:
                        q_num = int(match.group(1))
                        # Basic sanity check: Question number shouldn't be crazy huge relative to count
                        if q_num > 1000: continue 
                        is_match = True
                        break
                    except: pass
            
            if is_match:
                questions.append({
                    "q_num": q_num,
                    "page": p_idx,
                    "rect": fitz.Rect(b[0], b[1], b[2], b[3]), # The text block rect
                    "raw_text": text[:20] + "..."
                })

    # Sort final list by Question Number to ensure 1, 2, 3 sequence
    # This fixes cases where a header/footer might have been picked up erroneously
    # Or if layout sorting wasn't perfect
    questions.sort(key=lambda x: x["q_num"])
    
    return questions

def crop_and_stitch(job_id, doc, questions, export_dir, pdf_type):
    """
    Uses the precise Text Block coordinates to define Crop Areas.
    Stitches accurately from Q(n).top to Q(n+1).top
    """
    final_output = []
    
    # Universal Padding
    PAD_TOP = 15
    PAD_BOTTOM = 10
    
    for i, q in enumerate(questions):
        curr_p = q["page"]
        
        # Start: Top of current question block (minus padding)
        y_start = max(0, q["rect"].y0 - PAD_TOP)
        
        # Calculate End Point
        y_end = -1
        is_multipage = False
        
        # Look at next question to determine cut point
        if i + 1 < len(questions):
            next_q = questions[i+1]
            
            if next_q["page"] == curr_p:
                # Same Page: Cut slightly before next question starts
                # Logic check: Next Q should be below Current Q
                # For Double Col: Check if Next Q is in same column
                
                if pdf_type == 'double_col':
                    # Determine columns
                    curr_is_left = q["rect"].x0 < (doc[curr_p].rect.width / 2)
                    next_is_left = next_q["rect"].x0 < (doc[curr_p].rect.width / 2)
                    
                    if curr_is_left == next_is_left:
                        # Same column, standard cut
                        y_end = max(y_start + 50, next_q["rect"].y0 - PAD_BOTTOM)
                    else:
                        # Different column, Current Q goes to bottom of its column
                        y_end = doc[curr_p].rect.height - 40
                else:
                    # Single Col: Simple cut
                    y_end = max(y_start + 50, next_q["rect"].y0 - PAD_BOTTOM)
            else:
                # Next Q is on later page
                is_multipage = True
                y_end = doc[curr_p].rect.height - 40 # Margin
        else:
            # Last question
            y_end = doc[curr_p].rect.height - 40

        # Define Crop Width
        page_width = doc[curr_p].rect.width
        if pdf_type == 'double_col':
            # Check which side this question is on
            if q["rect"].x0 < (page_width / 2):
                x_start, x_end = 0, (page_width / 2)
            else:
                x_start, x_end = (page_width / 2), page_width
        else:
            # Single Col: Full Width
            x_start, x_end = 0, page_width

        segments = []
        
        # 1. Main Segment
        segments.append({
            "page": curr_p,
            "rect": fitz.Rect(x_start, y_start, x_end, y_end)
        })
        
        # 2. Stitching (Multi-page)
        if is_multipage and i + 1 < len(questions):
            next_q = questions[i+1]
            next_p = next_q["page"]
            
            # Simple stitching: Grab intermediate pages fully? 
            # Or assume same column logic? 
            # Safe bet: Grab Top of next page until Next Q starts.
            
            # Determine column of Next Q to know width
            if pdf_type == 'double_col':
                 if next_q["rect"].x0 < (doc[next_p].rect.width / 2):
                     nx_start, nx_end = 0, (doc[next_p].rect.width / 2)
                 else:
                     nx_start, nx_end = (doc[next_p].rect.width / 2), doc[next_p].rect.width
            else:
                nx_start, nx_end = 0, doc[next_p].rect.width
            
            next_q_y = max(50, next_q["rect"].y0 - PAD_BOTTOM)
            
            # Add segment from next page
            segments.append({
                "page": next_p,
                "rect": fitz.Rect(nx_start, 40, nx_end, next_q_y) # Start at 40 to skip header
            })

        # Render & Stitch
        images = []
        total_h = 0
        max_w = 0
        
        for seg in segments:
            page = doc[seg['page']]
            clip_rect = seg['rect'] & page.rect # Safe intersection
            
            if clip_rect.is_empty: continue
            
            try:
                pix = page.get_pixmap(dpi=200, clip=clip_rect)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                total_h += img.height
                max_w = max(max_w, img.width)
            except: pass
            
        if not images: continue
        
        # Final Image
        final_img = Image.new('RGB', (max_w, total_h), (255, 255, 255))
        cy = 0
        for img in images:
            final_img.paste(img, (0, cy))
            cy += img.height
            
        # Naming
        fname = f"Stark__--__{q['q_num']}__--__1.png"
        fpath = os.path.join(export_dir, fname)
        if os.path.exists(fpath):
            fname = f"Stark__--__{q['q_num']}_{uuid.uuid4().hex[:4]}__--__1.png"
            fpath = os.path.join(export_dir, fname)
            
        final_img.save(fpath)
        final_output.append({"label": str(q['q_num']), "filename": fname})
        
    return final_output

def worker_process(job_id, pdf_path, job_dir, pdf_type):
    with job_semaphore:
        export_dir = os.path.join(job_dir, "master_package")
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            doc = fitz.open(pdf_path)
            pdf_hash = get_pdf_hash(pdf_path)
            
            log_job(job_id, f"🚀 STARTING REGEX JOB ({pdf_type.upper()}). Pages: {len(doc)}", "INFO")
            
            # 1. Find Coordinates via Regex
            questions = find_questions_via_regex(doc, pdf_type)
            
            if not questions:
                log_job(job_id, "❌ No questions found via Text Layer. PDF might be scanned images only.", "ERROR")
                raise Exception("No text layer found. Please OCR your PDF first.")
                
            qs_found = [q['q_num'] for q in questions]
            log_job(job_id, f"✅ Regex found {len(questions)} questions: {qs_found[:10]}...", "SUCCESS")
            
            # 2. Crop
            log_job(job_id, "✂️ Precision Cropping...", "INFO")
            final_qs = crop_and_stitch(job_id, doc, questions, export_dir, pdf_type)
            
            # 3. Pack
            log_job(job_id, "📦 Generating Data Package...", "INFO")
            data_json = {
                "testConfig": {"pdfFileHash": pdf_hash},
                "pdfCropperData": {"Stark": {"Stark": {}}},
                "appVersion": "1.30.0",
                "generatedBy": "Team_Stark_Regex_V1"
            }
            
            for q in final_qs:
                key = q['label']
                if "_" in q['filename']:
                     try: key = q['filename'].split("__--__")[1]
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
            log_job(job_id, "✅ DONE.", "SUCCESS")
            
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            log_job(job_id, f"🔥 FATAL: {str(e)}", "ERROR")
        finally:
            if 'doc' in locals(): doc.close()
            try: shutil.rmtree(export_dir)
            except: pass
            gc.collect()

# --- API ROUTES ---

@app.route('/upload', methods=['POST'])
def start_job():
    if not is_authorized(request): return jsonify({"error": "Unauthorized"}), 403
    if 'pdf' not in request.files: return jsonify({"error": "No file"}), 400
    
    file = request.files['pdf']
    job_id = str(uuid.uuid4())[:8]
    pdf_type = request.form.get('pdf_type', 'single_col')
    
    job_dir = os.path.join(BASE_TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    save_path = os.path.join(job_dir, "source.pdf")
    file.save(save_path)
    
    jobs[job_id] = {"status": "queued", "logs": [], "file_path": None, "error": None}
    
    thread = threading.Thread(target=worker_process, args=(job_id, save_path, job_dir, pdf_type))
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
    return "Team Stark V33 (Regex Engine) 🚀"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
