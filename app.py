from flask import Flask, request, send_file, render_template
from flask_cors import CORS
from pathlib import Path
import easyocr
import cv2
import numpy as np
import re
import os

# --- Create required folders ---
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
STATIC_DIR = Path("static")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# --- Initialize Flask App ---
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# --- Initialize EasyOCR Reader ---
reader = easyocr.Reader(['en'], gpu=False, detect_network='craft', recog_network='standard')

# --- Helper Functions ---
def preprocess_image(file_path):
    """Preprocess image for terminal/log/URL text."""
    img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file or path")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert if terminal background is dark
    if np.mean(gray) < 127:
        gray = 255 - gray

    # Adaptive threshold for better text contrast
    gray = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)

    # Optional denoising
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    return gray

def is_code_like(word):
    """Return True if word contains numbers, symbols, or uppercase sequences (URLs, code, commands)."""
    return bool(re.search(r'[^A-Za-z]', word))

def ocr_easyocr(img):
    """Perform OCR with EasyOCR and return text + visualization image."""
    result1 = reader.readtext(img)
    result2 = reader.readtext(255 - img)

    # Choose result with more text
    chosen_result = result1 if len(" ".join([t[1] for t in result1])) > len(" ".join([t[1] for t in result2])) else result2

    # Sort top-to-bottom, left-to-right
    chosen_result.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    # Group into lines
    heights = [r[0][2][1] - r[0][0][1] for r in chosen_result]
    avg_height = np.mean(heights) if heights else 1
    Y_TOLERANCE = 0.6 * avg_height

    lines = []
    current_line = []
    line_y_ref = chosen_result[0][0][0][1] if chosen_result else 0

    for (bbox, text, _) in chosen_result:
        if abs(bbox[0][1] - line_y_ref) < Y_TOLERANCE:
            current_line.append(text)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [text]
            line_y_ref = bbox[0][1]
    if current_line:
        lines.append(" ".join(current_line))

    # Join lines preserving terminal/log formatting
    full_text = "\n".join([line.strip() for line in lines])

    # Draw bounding boxes
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (bbox, t, _) in chosen_result:
        tl = tuple(map(int, bbox[0]))
        br = tuple(map(int, bbox[2]))
        cv2.rectangle(vis_img, tl, br, (0, 255, 0), 2)

    return full_text, vis_img

def clean_text(text):
    """Clean text while preserving URLs, logs, code, and commands."""
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    # Preserve line breaks for terminal/logs
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        words = line.split()
        cleaned_words = []
        for w in words:
            if is_code_like(w):
                cleaned_words.append(w)  # leave URLs, commands, logs intact
            else:
                cleaned_words.append(w)
        cleaned_lines.append(" ".join(cleaned_words))
    cleaned_text = "\n".join(cleaned_lines)

    # Minor post-processing for punctuation spacing
    cleaned_text = re.sub(r'\s*([.,:;!?])', r'\1', cleaned_text)
    return cleaned_text.strip()

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return "No file part in the request", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = Path(file.filename).name
    file_path = UPLOAD_DIR / filename
    try:
        file.save(file_path)
        processed_img = preprocess_image(file_path)
        text, vis_img = ocr_easyocr(processed_img)
        cleaned_text = clean_text(text)

        # Save text result
        result_path = RESULT_DIR / f"{filename}.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Save visualization
        vis_path = RESULT_DIR / f"vis_{filename}"
        cv2.imwrite(str(vis_path), vis_img)

    except Exception as e:
        return f"OCR Failed due to a server error: {type(e).__name__}: {e}", 500

    return f"""
    <div class="status">
        <h3>✅ OCR Completed!</h3>
        <p><b>Original Filename:</b> {filename}</p>
        <p><b>Extracted Text:</b></p>
        <pre style="white-space: pre-wrap;">{cleaned_text}</pre>
        <p><b>Bounding Box Visualization:</b></p>
        <img src="/download_vis/{filename}" alt="Image with OCR bounding boxes"><br><br>
        <a href="/download/{filename}">Download Text File</a><br><br>
        <a href="/">Upload Another Image</a>
    </div>
    """

@app.route("/download/<filename>")
def download(filename):
    result_path = RESULT_DIR / f"{filename}.txt"
    if not result_path.exists():
        return "Text file not found", 404
    return send_file(result_path, as_attachment=True)

@app.route("/download_vis/<filename>")
def download_vis(filename):
    vis_path = RESULT_DIR / f"vis_{filename}"
    if not vis_path.exists():
        return "Visual file not found", 404
    return send_file(vis_path, mimetype='image/jpeg')

# --- Run App ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

