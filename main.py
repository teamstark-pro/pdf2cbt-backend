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
from PIL import Image, ImageOps
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
        print(f"[SYSTEM] ⚠️ Rate limit on key ...{key[-4:]}. Pausing.", flush=True)

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

# ==========================================
# PHASE 0: AI RECONNAISSANCE (Get Pattern)
# ==========================================

def get_dynamic_regex_pattern(job_id, doc):
    """
    Sends the first page to Groq to identify the exact question numbering format.
    Returns a Python-compatible Regex string.
    """
    try:
        page = doc[0] # Analyze Page 1
        pix = page.get_pixmap(dpi=100)
        b64 = pixmap_to_base64(pix)
        
        prompt = """
        Look at this exam page.
        Identify how the questions are numbered.
        
        Examples:
        - "1.", "2." -> Pattern: ^\s*(\d+)\.
        - "Q1", "Q2" -> Pattern: ^\s*Q(\d+)
        - "(1)", "(2)" -> Pattern: ^\s*\((\d+)\)
        - "Question 1:", "Question 2:" -> Pattern: ^\s*Question\s*(\d+)[:\.]
        - "1)", "2)" -> Pattern: ^\s*(\d+)\)
        
        TASK:
        Return the **Single Best Python Regex** to capture the question number.
        The regex MUST have a capturing group `(\d+)` for the number itself.
        
        OUTPUT JSON:
        { "pattern": "raw_python_regex_string", "confidence": "high" }
        """
        
        payload = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
        
        response = groq_manager.call_vision_batch(payload)
        
        if response and "pattern" in response:
            pat = response["pattern"]
            # Validate regex validity
            try:
                re.compile(pat)
                return pat
            except:
                log_job(job_id, f"⚠️ AI returned invalid regex: {pat}. Using fallback.", "WARN")
                return None
    except Exception as e:
        log_job(job_id, f"⚠️ AI Recon failed: {e}", "WARN")
        return None
    
    return None

# ==========================================
# STRATEGY 1: REGEX (Dynamic)
# ==========================================

def find_questions_via_regex(doc, pdf_type, dynamic_pattern=None):
    questions = []
    
    # Default Patterns (Fallback)
    patterns = [
        r"^\s*(\d+)[\.\)\-\:]",           # 1. or 1)
        r"^\s*Q(?:uestion)?[\.\s\-]*(\d+)", # Q1
        r"^\s*\[(\d+)\]",                   # [1]
        r"^\s*Problem\s*(\d+)"              # Problem 1
    ]
    
    # If AI gave a pattern, prioritize it!
    if dynamic_pattern:
        log_job("SYSTEM", f"🎯 Using AI-Detected Regex: {dynamic_pattern}", "INFO")
        patterns.insert(0, dynamic_pattern)
        
    bad_keywords = ["Solution", "Answer", "Exp:", "Explanation", "Page", "Fig", "Table"]
    
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        width = page.rect.width
        blocks = page.get_text("blocks")
        
        # Sort blocks to prevent zig-zag
        if pdf_type == 'double_col':
            midpoint = width / 2
            left_col = [b for b in blocks if b[0] < midpoint]
            right_col = [b for b in blocks if b[0] >= midpoint]
            left_col.sort(key=lambda b: b[1])
            right_col.sort(key=lambda b: b[1])
            sorted_blocks = left_col + right_col
        else:
            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        for b in sorted_blocks:
            text = b[4].strip()
            if not text: continue
            if any(bk in text for bk in bad_keywords): continue
            
            for pat in patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    try:
                        q_num = int(match.group(1))
                        if q_num > 500 and len(questions) < 10: continue 
                        questions.append({
                            "q_num": q_num,
                            "page": p_idx,
                            "rect": fitz.Rect(b[0], b[1], b[2], b[3])
                        })
                        break # Found a match for this block, move to next block
                    except: pass

    if not questions: return None
    questions.sort(key=lambda x: x["q_num"])
    
    # Strict Validation Sequence
    valid_qs = []
    last_q = 0
    strikes = 0
    
    for q in questions:
        if q["q_num"] <= last_q: continue
        if q["q_num"] > last_q + 10: strikes += 1
        valid_qs.append(q)
        last_q = q["q_num"]
        
    if strikes > 3 or len(valid_qs) < 3: 
        return None 
        
    return valid_qs

def crop_and_stitch_regex(job_id, doc, questions, export_dir, pdf_type):
    final_output = []
    PAD_TOP = 15
    PAD_BOTTOM = 10
    
    for i, q in enumerate(questions):
        curr_p = q["page"]
        y_start = max(0, q["rect"].y0 - PAD_TOP)
        y_end = -1
        is_multipage = False
        
        if i + 1 < len(questions):
            next_q = questions[i+1]
            if next_q["page"] == curr_p:
                if pdf_type == 'double_col':
                    curr_is_left = q["rect"].x0 < (doc[curr_p].rect.width / 2)
                    next_is_left = next_q["rect"].x0 < (doc[curr_p].rect.width / 2)
                    if curr_is_left == next_is_left:
                        y_end = max(y_start + 50, next_q["rect"].y0 - PAD_BOTTOM)
                    else:
                        y_end = doc[curr_p].rect.height - 40
                else:
                    y_end = max(y_start + 50, next_q["rect"].y0 - PAD_BOTTOM)
            else:
                is_multipage = True
                y_end = doc[curr_p].rect.height - 40
        else:
            y_end = doc[curr_p].rect.height - 40

        page_width = doc[curr_p].rect.width
        if pdf_type == 'double_col':
            if q["rect"].x0 < (page_width / 2):
                x_start, x_end = 0, (page_width / 2)
            else:
                x_start, x_end = (page_width / 2), page_width
        else:
            x_start, x_end = 0, page_width

        segments = [{"page": curr_p, "rect": fitz.Rect(x_start, y_start, x_end, y_end)}]
        
        if is_multipage and i + 1 < len(questions):
            next_q = questions[i+1]
            next_p = next_q["page"]
            if pdf_type == 'double_col':
                 if next_q["rect"].x0 < (doc[next_p].rect.width / 2):
                     nx_start, nx_end = 0, (doc[next_p].rect.width / 2)
                 else:
                     nx_start, nx_end = (doc[next_p].rect.width / 2), doc[next_p].rect.width
            else:
                nx_start, nx_end = 0, doc[next_p].rect.width
            
            next_q_y = max(50, next_q["rect"].y0 - PAD_BOTTOM)
            segments.append({"page": next_p, "rect": fitz.Rect(nx_start, 40, nx_end, next_q_y)})

        images = []
        total_h = 0
        max_w = 0
        for seg in segments:
            page = doc[seg['page']]
            clip_rect = seg['rect'] & page.rect
            if clip_rect.is_empty: continue
            try:
                pix = page.get_pixmap(dpi=200, clip=clip_rect)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                total_h += img.height
                max_w = max(max_w, img.width)
            except: pass
            
        if not images: continue
        final_img = Image.new('RGB', (max_w, total_h), (255, 255, 255))
        cy = 0
        for img in images:
            final_img.paste(img, (0, cy))
            cy += img.height
            
        fname = f"Stark__--__{q['q_num']}__--__1.png"
        fpath = os.path.join(export_dir, fname)
        if os.path.exists(fpath):
            fname = f"Stark__--__{q['q_num']}_{uuid.uuid4().hex[:4]}__--__1.png"
            fpath = os.path.join(export_dir, fname)
        final_img.save(fpath)
        final_output.append({"label": str(q['q_num']), "filename": fname})
        
    return final_output

# ==========================================
# STRATEGY 2: VISION AI (Fallback)
# ==========================================

def get_questions_ai_coordinates(job_id, doc, page_indices, pdf_type):
    base_text = """
    Analyze this exam page.
    Identify the FULL BOUNDING BOX for every question (Number + Text + Options).
    
    Output JSON: "img_0": [{ "q_num": Int, "x_start": Int(0-1000), "y_start": Int(0-1000), "x_end": Int(0-1000), "y_end": Int(0-1000) }]
    """
    if pdf_type == 'double_col':
        base_text += "\nLayout: DOUBLE COLUMN. Scan Left column, then Right column."
    else:
        base_text += "\nLayout: SINGLE COLUMN. Scan Top to Bottom."

    payload_content = [{"type": "text", "text": base_text}]
    valid_map = {}
    
    for i, p_idx in enumerate(page_indices):
        try:
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=100)
            b64 = pixmap_to_base64(pix)
            if not b64: continue
            key = f"img_{i}"
            valid_map[key] = p_idx
            payload_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
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
                            if x1 <= x0: x1 = 1000
                            if y1 <= y0: y1 = y0 + 50
                            cleaned_items.append({"q": q_num, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
                        except: pass
                if pdf_type == 'double_col':
                    cleaned_items.sort(key=lambda x: (0 if x["x0"] < 500 else 1, x["y0"]))
                else:
                    cleaned_items.sort(key=lambda x: x["y0"])
                result_map[real_page_idx] = cleaned_items
    return result_map

def extract_and_stitch_vision(job_id, doc, vision_map, export_dir, pdf_type):
    all_qs_coords = []
    for p_idx, items in vision_map.items():
        page = doc[p_idx]
        w = page.rect.width
        h = page.rect.height
        for item in items:
            if pdf_type == 'single_col':
                if item['x0'] < 100: item['x0'] = 0
                if item['x1'] > 900: item['x1'] = 1000
            elif pdf_type == 'double_col':
                if item['x0'] < 20: item['x0'] = 0
                if item['x1'] > 980: item['x1'] = 1000
            
            x0 = (item['x0'] / 1000.0) * w
            y0 = (item['y0'] / 1000.0) * h
            x1 = (item['x1'] / 1000.0) * w
            y1 = (item['y1'] / 1000.0) * h
            all_qs_coords.append({
                "label": str(item['q']),
                "q_num_int": item['q'],
                "page": p_idx,
                "rect": fitz.Rect(x0, y0, x1, y1),
                "raw_y1_score": item['y1']
            })

    all_qs_coords.sort(key=lambda x: x["q_num_int"])
    if not all_qs_coords: return []

    final_output = []
    PAD = 20
    for i, q in enumerate(all_qs_coords):
        curr_p = q["page"]
        rect = q["rect"]
        safe_rect = fitz.Rect(max(0, rect.x0 - PAD), max(0, rect.y0 - PAD), min(doc[curr_p].rect.width, rect.x1 + PAD), min(doc[curr_p].rect.height, rect.y1 + PAD))
        clip_rect = safe_rect & doc[curr_p].rect
        if clip_rect.is_empty: continue
        pix = doc[curr_p].get_pixmap(dpi=200, clip=clip_rect)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        fname = f"Stark__--__{q['label']}__--__1.png"
        fpath = os.path.join(export_dir, fname)
        if os.path.exists(fpath):
            fname = f"Stark__--__{q['label']}_{uuid.uuid4().hex[:4]}__--__1.png"
            fpath = os.path.join(export_dir, fname)
        img.save(fpath)
        final_output.append({"label": q['label'], "filename": fname})
    return final_output

# ==========================================
# STRATEGY 3: PIXEL LAYOUT (Scanned Fallback)
# ==========================================

def analyze_pixel_layout(job_id, doc, export_dir, pdf_type):
    final_output = []
    question_counter = 1
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        pix = page.get_pixmap(dpi=72)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        gray = ImageOps.grayscale(img)
        width, height = gray.size
        pixels = gray.load()
        INK_THRESH = 200 
        
        def get_segments(x_start, x_end):
            segments = []
            in_block = False
            start_y = 0
            for y in range(height):
                has_ink = False
                for x in range(int(x_start), int(x_end), 2):
                    if pixels[x, y] < INK_THRESH:
                        has_ink = True
                        break
                if has_ink and not in_block:
                    in_block = True
                    start_y = y
                elif not has_ink and in_block:
                    is_gap = True
                    for next_y in range(y, min(y + 10, height)):
                        for next_x in range(int(x_start), int(x_end), 2):
                            if pixels[next_x, next_y] < INK_THRESH:
                                is_gap = False
                                break
                        if not is_gap: break
                    if is_gap:
                        in_block = False
                        if (y - start_y) > 30: segments.append((start_y, y))
            return segments

        if pdf_type == 'double_col':
            mid = width / 2
            col1_segs = get_segments(0, mid)
            col2_segs = get_segments(mid, width)
            for (y0, y1) in col1_segs:
                scale_y = page.rect.height / height
                scale_x = page.rect.width / width
                rect = fitz.Rect(0, y0 * scale_y, mid * scale_x, y1 * scale_y)
                final_output.append({"rect": rect, "page": p_idx})
            for (y0, y1) in col2_segs:
                scale_y = page.rect.height / height
                scale_x = page.rect.width / width
                rect = fitz.Rect(mid * scale_x, y0 * scale_y, width * scale_x, y1 * scale_y)
                final_output.append({"rect": rect, "page": p_idx})
        else: 
            segs = get_segments(0, width)
            for (y0, y1) in segs:
                scale_y = page.rect.height / height
                rect = fitz.Rect(0, y0 * scale_y, page.rect.width, y1 * scale_y)
                final_output.append({"rect": rect, "page": p_idx})

    processed_files = []
    for item in final_output:
        page = doc[item["page"]]
        rect = item["rect"]
        clip_rect = fitz.Rect(rect.x0, max(0, rect.y0 - 10), rect.x1, min(page.rect.height, rect.y1 + 10))
        pix = page.get_pixmap(dpi=200, clip=clip_rect)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        fname = f"Stark__--__{question_counter}__--__1.png"
        fpath = os.path.join(export_dir, fname)
        img.save(fpath)
        processed_files.append({"label": str(question_counter), "filename": fname})
        question_counter += 1
    return processed_files

# ==========================================
# WORKER PROCESS
# ==========================================

def worker_process(job_id, pdf_path, job_dir, pdf_type):
    with job_semaphore:
        export_dir = os.path.join(job_dir, "master_package")
        os.makedirs(export_dir, exist_ok=True)
        try:
            doc = fitz.open(pdf_path)
            pdf_hash = get_pdf_hash(pdf_path)
            log_job(job_id, f"🚀 JOB STARTED ({pdf_type}). Pages: {len(doc)}", "INFO")
            
            final_qs = []
            
            # PHASE 0: RECONNAISSANCE (Smart Regex)
            log_job(job_id, "🛰️ AI Recon: Detecting Numbering Pattern...", "INFO")
            dynamic_pattern = get_dynamic_regex_pattern(job_id, doc)
            
            # PHASE 1: REGEX SCAN
            log_job(job_id, "🔍 Strategy 1: Text Layer (Smart Regex)...", "INFO")
            questions = find_questions_via_regex(doc, pdf_type, dynamic_pattern)
            
            if questions:
                log_job(job_id, f"✅ Smart Regex found {len(questions)} items.", "SUCCESS")
                final_qs = crop_and_stitch_regex(job_id, doc, questions, export_dir, pdf_type)
            
            # PHASE 2: VISION AI (Fallback)
            if not final_qs:
                log_job(job_id, "⚠️ Regex failed. Strategy 2: Vision AI...", "WARN")
                FULL_VISION_DATA = {}
                BATCH_SIZE = 2
                batches = [list(range(i, min(i + BATCH_SIZE, len(doc)))) for i in range(0, len(doc), BATCH_SIZE)]
                for indices in batches:
                    log_job(job_id, f"   -> AI Scanning Pages {[p+1 for p in indices]}...", "INFO")
                    batch_doc = fitz.open(pdf_path) 
                    try:
                        res = get_questions_ai_coordinates(job_id, batch_doc, indices, pdf_type)
                        if res: 
                            for k,v in res.items(): FULL_VISION_DATA[k] = v
                        time.sleep(1)
                    except: pass
                    finally: batch_doc.close()
                if FULL_VISION_DATA:
                    log_job(job_id, "✂️ Vision Cropping...", "INFO")
                    main_doc = fitz.open(pdf_path)
                    final_qs = extract_and_stitch_vision(job_id, main_doc, FULL_VISION_DATA, export_dir, pdf_type)
                    main_doc.close()

            # PHASE 3: PIXEL LAYOUT (Last Resort)
            if not final_qs:
                log_job(job_id, "⚠️ Vision AI failed. Strategy 3: Pixel Layout...", "WARN")
                main_doc = fitz.open(pdf_path)
                final_qs = analyze_pixel_layout(job_id, main_doc, export_dir, pdf_type)
                main_doc.close()
                if final_qs:
                    log_job(job_id, f"✅ Pixel Scan found {len(final_qs)} blocks.", "SUCCESS")

            if not final_qs:
                raise Exception("All strategies failed.")
                
            log_job(job_id, "📦 Packaging...", "INFO")
            data_json = {
                "testConfig": {"pdfFileHash": pdf_hash},
                "pdfCropperData": {"Stark": {"Stark": {}}},
                "appVersion": "1.30.0",
                "generatedBy": "Team_Stark_Smart_V33"
            }
            for q in final_qs:
                key = q['label']
                if "_" in q['filename']:
                     try: key = q['filename'].split("__--__")[1]
                     except: pass
                data_json["pdfCropperData"]["Stark"]["Stark"][key] = {
                    "que": key, "type": "mcq", "marks": {"cm": 4, "im": -1},
                    "answerOptions": "4", "pdfData": [{"x1": 5, "x2": 995, "y1": 100, "y2": 500, "page": 1}]
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
            log_job(job_id, "✅ JOB COMPLETE.", "SUCCESS")
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
    return "Team Stark V33 (Smart Regex + Fallback) 🚀"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, threaded=True)
