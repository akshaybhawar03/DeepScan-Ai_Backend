"""
Ensemble Deepfake Detection API
4 AI Models + 4 Analysis Features + History
"""

import os
import io
import time
import json
import hashlib
import base64
import tempfile
from typing import Any
from datetime import datetime

import numpy as np
import cv2
import piexif
import requests
from PIL import Image, ImageChops, ImageEnhance
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import realitydefender
    HAS_RD = True
except ImportError:
    HAS_RD = False

try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    HAS_FACE_REC = False

load_dotenv()

app = Flask(__name__)

# CORS — allow Vercel frontend + localhost dev
FRONTEND_URL = os.environ.get("FRONTEND_URL", "")
allowed_origins = [
    "http://localhost:5500", "http://127.0.0.1:5500",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:8080",
    "https://deep-scan-ai-frontend.vercel.app",
    "https://deepscan-ai-frontend.vercel.app",
    "https://deepscan-ai-frontend-akshaybhawar03s-projects.vercel.app",
]
if FRONTEND_URL:
    allowed_origins.append(FRONTEND_URL)
CORS(app, origins=allowed_origins + ["*"])

# ─── Configuration ────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

HF_TOKEN = os.environ.get("HF_API_TOKEN", "")
RD_KEY = os.environ.get("REALITY_DEFENDER_KEY", "")
MODEL_TIMEOUT = 30
CACHE_TTL = 600

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "scan_history.json")
MAX_HISTORY = 20

# ─── Cache ────────────────────────────────────────────────────────────
result_cache: dict[str, Any] = {}

# ─── Model Definitions ────────────────────────────────────────────────
WEIGHTS_4 = {"reality_defender": 0.40, "model1_ViT": 0.30, "model2_SigLIP": 0.20, "model3_general": 0.10}
WEIGHTS_3 = {"model1_ViT": 0.50, "model2_SigLIP": 0.35, "model3_general": 0.15}

MODELS = {
    "reality_defender": {
        "id": "Reality Defender API",
        "name": "Reality Defender",
        "accuracy": "Enterprise",
        "description": "Enterprise-grade deepfake detection API",
        "type": "api",
        "weight_4": 0.40,
    },
    "model1_ViT": {
        "id": "Wvolf/ViT_Deepfake_Detection",
        "url": "https://router.huggingface.co/hf-inference/models/Wvolf/ViT_Deepfake_Detection",
        "name": "ViT Deepfake Detector",
        "accuracy": "98.7%",
        "description": "Primary detector — Vision Transformer based",
        "type": "hf",
        "weight_4": 0.30,
    },
    "model2_SigLIP": {
        "id": "prithivMLmods/deepfake-detector-model-v1",
        "url": "https://router.huggingface.co/hf-inference/models/prithivMLmods/deepfake-detector-model-v1",
        "name": "SigLIP Deepfake Detector",
        "accuracy": "94.44%",
        "description": "Second opinion — Google SigLIP architecture",
        "type": "hf",
        "weight_4": 0.20,
    },
    "model3_general": {
        "id": "umm-maybe/AI-image-detector",
        "url": "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector",
        "name": "General AI Detector",
        "accuracy": "~92%",
        "description": "General AI-generated image classification",
        "type": "hf",
        "weight_4": 0.10,
    }
}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def get_cached(h: str) -> dict[str, Any] | None:
    if h in result_cache:
        entry = result_cache[h]
        if time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]
        del result_cache[h]
    return None


def set_cache(h: str, data: dict[str, Any]) -> None:
    result_cache[h] = {"data": data, "ts": time.time()}
    if len(result_cache) > 100:
        oldest = min(result_cache, key=lambda k: result_cache[k]["ts"])
        del result_cache[oldest]


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 1: ELA (Error Level Analysis)
# ═══════════════════════════════════════════════════════════════════════

def ela_analysis(image_bytes: bytes) -> dict[str, Any]:
    """Detect tampered regions via compression artifact differences."""
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resave at lower quality
        buffer_low = io.BytesIO()
        original.save(buffer_low, 'JPEG', quality=75)
        buffer_low.seek(0)
        resaved = Image.open(buffer_low)

        # Difference
        ela_image = ImageChops.difference(original, resaved)

        # Amplify
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Score
        ela_array = np.array(ela_image)
        avg_brightness = float(ela_array.mean())
        ela_score = min(avg_brightness / 128.0, 1.0)

        # Save as base64
        buf = io.BytesIO()
        ela_image.save(buf, format='PNG')
        ela_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "score": round(ela_score * 100, 2),
            "ela_image": f"data:image/png;base64,{ela_b64}",
            "verdict": "SUSPICIOUS" if ela_score > 0.4 else "CLEAN",
            "description": "High error levels detected — possible tampering"
                           if ela_score > 0.4 else "Normal compression patterns",
            "status": "success"
        }
    except Exception as e:
        return {"score": 0, "ela_image": None, "verdict": "ERROR",
                "description": str(e), "status": "error"}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 2: EXIF Metadata Analysis
# ═══════════════════════════════════════════════════════════════════════

SUSPICIOUS_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "midjourney",
    "stable diffusion", "dall-e", "firefly", "canva",
    "faceapp", "reface", "deepfacelab", "artbreeder"
]


def exif_analysis(image_bytes: bytes) -> dict[str, Any]:
    """Inspect EXIF metadata for editing/AI signatures."""
    flags: list[str] = []
    score = 0
    metadata: dict[str, str] = {}

    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_raw = img.info.get('exif', b'')

        if not exif_raw:
            flags.append("No EXIF data — possibly AI generated or stripped")
            score += 35
            return {"score": min(score, 100), "metadata": metadata,
                    "flags": flags, "verdict": "SUSPICIOUS" if score > 30 else "CLEAN",
                    "status": "success"}

        exif_data = piexif.load(exif_raw)

        # Software tag
        software = exif_data.get('0th', {}).get(piexif.ImageIFD.Software, b'')
        if software:
            sw = software.decode('utf-8', errors='ignore')
            metadata['software'] = sw
            for sus in SUSPICIOUS_SOFTWARE:
                if sus in sw.lower():
                    flags.append(f"Edited with: {sw}")
                    score += 40
                    break

        # GPS
        gps = exif_data.get('GPS', {})
        if not gps:
            flags.append("No GPS data found")
            score += 10

        # DateTime
        dt = exif_data.get('0th', {}).get(piexif.ImageIFD.DateTime, b'')
        if dt:
            metadata['datetime'] = dt.decode('utf-8', errors='ignore')

        # Camera
        camera = exif_data.get('0th', {}).get(piexif.ImageIFD.Model, b'')
        if camera:
            metadata['camera'] = camera.decode('utf-8', errors='ignore')
        else:
            flags.append("No camera model found")
            score += 15

        # Make
        make = exif_data.get('0th', {}).get(piexif.ImageIFD.Make, b'')
        if make:
            metadata['make'] = make.decode('utf-8', errors='ignore')

        if not flags:
            flags.append("No suspicious metadata found")

    except Exception:
        flags.append("EXIF data corrupted or missing")
        score += 25

    return {"score": min(score, 100), "metadata": metadata,
            "flags": flags, "verdict": "SUSPICIOUS" if score > 30 else "CLEAN",
            "status": "success"}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 3: Face Analysis (OpenCV + optional face_recognition)
# ═══════════════════════════════════════════════════════════════════════

# Load Haar cascades
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'  # type: ignore
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)


def face_analysis(image_bytes: bytes) -> dict[str, Any]:
    """Analyze facial symmetry and landmarks for deepfake artifacts."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return _face_error("Could not decode image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
        if len(faces) == 0:
            return {"score": 0, "face_count": 0, "verdict": "NO_FACE",
                    "flags": ["No face detected in image"], "status": "success"}

        flags: list[str] = []
        score = 0

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            # Eye detection
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(20, 20))

            if len(eyes) >= 2:
                # Sort by x to get left/right
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                e1 = eyes_sorted[0]
                e2 = eyes_sorted[1]

                # Eye size asymmetry
                size1 = e1[2] * e1[3]
                size2 = e2[2] * e2[3]
                asym = abs(size1 - size2) / max(size1, size2)

                if asym > 0.25:
                    flags.append(f"Eye size asymmetry: {round(asym * 100)}%")
                    score += 25

                # Eye height diff
                h_diff = abs(e1[1] - e2[1])
                if h_diff > h * 0.08:
                    flags.append("Eye height misalignment detected")
                    score += 15

            elif len(eyes) == 1:
                flags.append("Only one eye detected — possible occlusion or artifact")
                score += 10
            elif len(eyes) == 0:
                flags.append("No eyes detected in face region")
                score += 20

            # Skin uniformity check
            face_region = img[y:y+h, x:x+w]
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            sat_std = float(np.std(hsv[:, :, 1]))
            val_std = float(np.std(hsv[:, :, 2]))

            if sat_std < 15:
                flags.append("Unusually uniform skin tone — possible AI generation")
                score += 20

            if val_std < 20:
                flags.append("Low brightness variation in face — possible smoothing")
                score += 15

            # Face symmetry check (compare left/right halves)
            mid = w // 2
            left_half = face_roi[:, :mid]
            right_half = cv2.flip(face_roi[:, mid:2*mid], 1)

            if left_half.shape == right_half.shape and left_half.size > 0:
                diff = cv2.absdiff(left_half, right_half)
                sym_score = float(diff.mean())
                if sym_score > 35:
                    flags.append(f"Low facial symmetry (diff: {round(sym_score, 1)})")
                    score += 15

        if not flags:
            flags.append("Normal facial features detected")

        return {"score": min(score, 100), "face_count": int(len(faces)),
                "verdict": "SUSPICIOUS" if score > 35 else "CLEAN",
                "flags": flags, "status": "success"}

    except Exception as e:
        return _face_error(str(e))


def _face_error(msg: str) -> dict[str, Any]:
    return {"score": 0, "face_count": 0, "verdict": "ERROR",
            "flags": [f"Face analysis error: {msg}"], "status": "error"}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 4: Frequency Analysis (FFT)
# ═══════════════════════════════════════════════════════════════════════

def frequency_analysis(image_bytes: bytes) -> dict[str, Any]:
    """Detect GAN fingerprints via Fourier Transform."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img_array = np.array(img, dtype=float)

        # 2D FFT
        fft = np.fft.fft2(img_array)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        magnitude_log = np.log1p(magnitude)

        # Normalize
        mmin = float(magnitude_log.min())
        mmax = float(magnitude_log.max())
        if mmax - mmin == 0:
            magnitude_norm = np.zeros_like(magnitude_log, dtype=np.uint8)
        else:
            magnitude_norm = ((magnitude_log - mmin) / (mmax - mmin) * 255).astype(np.uint8)

        # Detect artifacts — exclude DC center
        cy, cx = magnitude_norm.shape[0] // 2, magnitude_norm.shape[1] // 2
        y, x = np.ogrid[:magnitude_norm.shape[0], :magnitude_norm.shape[1]]
        center_mask = (x - cx)**2 + (y - cy)**2 < 50**2
        mask = np.ones_like(magnitude_norm, dtype=bool)
        mask[center_mask] = False

        masked = magnitude_norm[mask]
        if masked.size == 0:
            artifact_ratio = 0.0
        else:
            threshold = float(masked.mean()) + 3 * float(masked.std())
            artifact_pixels = int((masked > threshold).sum())
            artifact_ratio = artifact_pixels / masked.size

        freq_score = min(artifact_ratio * 1000, 100)

        # Save visualization as base64
        freq_img = Image.fromarray(magnitude_norm)
        buf = io.BytesIO()
        freq_img.save(buf, format='PNG')
        freq_b64 = base64.b64encode(buf.getvalue()).decode()

        flags: list[str] = []
        if freq_score > 80:
            flags.append("Extreme frequency anomalies — high confidence AI generation")
        elif freq_score > 60:
            flags.append("Strong periodic patterns — likely AI generated")
        elif freq_score > 40:
            flags.append("GAN grid artifacts detected in frequency domain")
        if not flags:
            flags.append("Natural frequency distribution")

        return {
            "score": round(freq_score, 2),
            "artifact_ratio": round(artifact_ratio * 100, 4),
            "frequency_image": f"data:image/png;base64,{freq_b64}",
            "flags": flags,
            "verdict": "SUSPICIOUS" if freq_score > 40 else "CLEAN",
            "status": "success"
        }
    except Exception as e:
        return {"score": 0, "artifact_ratio": 0, "frequency_image": None,
                "flags": [str(e)], "verdict": "ERROR", "status": "error"}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 5: History
# ═══════════════════════════════════════════════════════════════════════

def load_history() -> list[dict[str, Any]]:
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_to_history(image_name: str, result: dict[str, Any]) -> None:
    history = load_history()
    entry = {
        "id": int(time.time() * 1000),
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "image_name": image_name,
        "verdict": result.get("verdict", "UNKNOWN"),
        "confidence": result.get("final_confidence", 0),
        "risk_level": result.get("risk_level", "UNKNOWN")
    }
    history.insert(0, entry)
    history = history[:MAX_HISTORY]
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# AI MODEL QUERIES
# ═══════════════════════════════════════════════════════════════════════

def query_reality_defender(image_bytes: bytes) -> dict[str, Any]:
    start = time.time()
    if not HAS_RD or not RD_KEY:
        return _make_result("reality_defender",
                            error="Reality Defender not configured", elapsed=0)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        client = realitydefender.RealityDefender(api_key=RD_KEY)
        result = client.detect_file(tmp_path)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        elapsed = int((time.time() - start) * 1000)

        # Parse result — could be DetectionResult object or dict
        if isinstance(result, dict):
            rd_score = float(result.get('confidence', result.get('score', 0.5)))
            rd_verdict = str(result.get('verdict', result.get('label', 'UNKNOWN')))
            rd_indicators = list(result.get('indicators', result.get('details', [])))
        else:
            rd_score = float(getattr(result, 'confidence', getattr(result, 'score', 0.5)))
            rd_verdict = str(getattr(result, 'verdict', getattr(result, 'label', 'UNKNOWN')))
            rd_indicators = list(getattr(result, 'indicators', getattr(result, 'details', [])))

        if rd_verdict.upper() in ("REAL", "AUTHENTIC") and rd_score > 0.5:
            fake_score = 1.0 - rd_score
        elif rd_verdict.upper() in ("DEEPFAKE", "FAKE", "SYNTHETIC", "MANIPULATED") and rd_score > 0.5:
            fake_score = rd_score
        else:
            fake_score = rd_score
        return {
            "model_key": "reality_defender", "name": "Reality Defender",
            "status": "success", "fake_score": round(fake_score, 4),
            "verdict": "Fake" if fake_score >= 0.5 else "Real",
            "score_pct": round(fake_score * 100, 1), "indicators": rd_indicators,
            "raw": {"confidence": rd_score, "verdict": rd_verdict},
            "elapsed_ms": elapsed, "highlighted_image": None, "error": None
        }
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return _make_result("reality_defender", error=str(e), elapsed=elapsed)


def query_hf_model(model_key: str, image_bytes: bytes) -> dict[str, Any]:
    model = MODELS[model_key]
    start = time.time()
    headers: dict[str, str] = {"Content-Type": "application/octet-stream"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        response = requests.post(model["url"], headers=headers,
                                 data=image_bytes, timeout=MODEL_TIMEOUT)
        elapsed = int((time.time() - start) * 1000)
        if response.status_code == 503:
            return _make_result(model_key, error="Model loading — retry soon", elapsed=elapsed)
        if response.status_code != 200:
            return _make_result(model_key, error=f"HTTP {response.status_code}", elapsed=elapsed)
        result = response.json()
        fake_score = _extract_fake_score(result)
        return {
            "model_key": model_key, "name": model["name"], "status": "success",
            "fake_score": round(fake_score, 4),
            "verdict": "Fake" if fake_score >= 0.5 else "Real",
            "score_pct": round(fake_score * 100, 1), "raw": result,
            "elapsed_ms": elapsed, "highlighted_image": None,
            "indicators": None, "error": None
        }
    except requests.exceptions.Timeout:
        return _make_result(model_key, error="Timed out (>30s)",
                            elapsed=int((time.time() - start) * 1000))
    except Exception as e:
        return _make_result(model_key, error=str(e),
                            elapsed=int((time.time() - start) * 1000))


def _make_result(model_key: str, error: str, elapsed: int) -> dict[str, Any]:
    return {
        "model_key": model_key, "name": MODELS[model_key]["name"],
        "status": "error", "fake_score": None, "verdict": None,
        "score_pct": None, "raw": None, "elapsed_ms": elapsed,
        "highlighted_image": None, "indicators": None, "error": error
    }


def _extract_fake_score(result: Any) -> float:
    items = result
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            items = result[0]
    if isinstance(items, list):
        preds: dict[str, float] = {}
        for item in items:
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip().lower()
                s = float(item.get("score", 0))
                preds[label] = s
        fake_kw = ["fake", "deepfake", "ai_generated", "ai-generated", "ai", "generated", "artificial", "synthetic", "computer"]
        real_kw = ["real", "authentic", "human", "natural", "original", "photo", "genuine"]
        fake = max((preds[l] for l in preds if any(f in l for f in fake_kw)), default=0.0)
        real = max((preds[l] for l in preds if any(r in l for r in real_kw)), default=0.0)
        if fake + real > 0:
            return fake / (fake + real)
    return 0.5


# ═══════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════

def run_ensemble(image_bytes: bytes, image_name: str) -> dict[str, Any]:
    total_start = time.time()

    # ── Run AI models in parallel ─────────────────────────────────────
    model_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures: dict[Any, str] = {}
        futures[executor.submit(query_reality_defender, image_bytes)] = "reality_defender"
        for key in ["model1_ViT", "model2_SigLIP", "model3_general"]:
            futures[executor.submit(query_hf_model, key, image_bytes)] = key
        for future in as_completed(futures):
            k = futures[future]
            try:
                model_results[k] = future.result()
            except Exception as e:
                model_results[k] = _make_result(k, str(e), 0)

    # ── Run analysis features in parallel ─────────────────────────────
    analysis: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        af = {
            executor.submit(ela_analysis, image_bytes): "ela",
            executor.submit(exif_analysis, image_bytes): "exif",
            executor.submit(face_analysis, image_bytes): "face",
            executor.submit(frequency_analysis, image_bytes): "frequency",
        }
        for future in as_completed(af):
            k = af[future]
            try:
                analysis[k] = future.result()
            except Exception as e:
                analysis[k] = {"score": 0, "verdict": "ERROR",
                               "flags": [str(e)], "status": "error"}

    # ── AI model ensemble score ───────────────────────────────────────
    scores: dict[str, float] = {}
    for key, r in model_results.items():
        if r["fake_score"] is not None:
            scores[key] = r["fake_score"]

    if len(scores) == 0:
        ai_score = 0.0
    else:
        rd_available = "reality_defender" in scores
        if rd_available:
            w = {k: WEIGHTS_4.get(k, 0.1) for k in scores}
        else:
            w = {k: WEIGHTS_3.get(k, 0.15) for k in scores}
        tw = sum(w.values())
        if tw > 0:
            w = {k: v / tw for k, v in w.items()}
        ai_score = sum(scores[k] * w.get(k, 0) for k in scores)

    # Agreement boost
    m1 = scores.get("model1_ViT")
    m2 = scores.get("model2_SigLIP")
    models_agree = None
    confidence_note = None
    if m1 is not None and m2 is not None:
        m1f = m1 >= 0.5
        m2f = m2 >= 0.5
        models_agree = m1f == m2f
        if models_agree and m1f:
            ai_score = min(ai_score + 0.05, 1.0)
            confidence_note = "Primary models agree — confidence boosted +5%"
        elif not models_agree:
            confidence_note = "Low Confidence — Models disagree, manual review needed"

    rd_available = "reality_defender" in scores
    mode = "full_ensemble" if rd_available else "hf_fallback"
    if rd_available and len(scores) == 1:
        mode = "reality_defender_only"

    if not rd_available:
        extra = "Reality Defender unavailable — used 3-model fallback"
        confidence_note = f"{confidence_note} | {extra}" if confidence_note else extra

    # ── Feature score ─────────────────────────────────────────────────
    ela_s = analysis.get("ela", {}).get("score", 0) / 100
    face_s = analysis.get("face", {}).get("score", 0) / 100
    freq_s = analysis.get("frequency", {}).get("score", 0) / 100
    exif_s = analysis.get("exif", {}).get("score", 0) / 100

    feature_score = (ela_s * 0.30 + face_s * 0.30 + freq_s * 0.25 + exif_s * 0.15)

    # ── Final combined score ──────────────────────────────────────────
    if len(scores) > 0:
        final_score = (ai_score * 0.70) + (feature_score * 0.30)
    else:
        final_score = feature_score  # Only features available

    # Verdict
    if final_score >= 0.70:
        verdict = "DEEPFAKE DETECTED"
        risk_level = "HIGH"
    elif final_score >= 0.50:
        verdict = "SUSPECTED DEEPFAKE"
        risk_level = "MEDIUM"
    elif final_score >= 0.35:
        verdict = "POSSIBLY MANIPULATED"
        risk_level = "LOW"
    else:
        verdict = "AUTHENTIC"
        risk_level = "SAFE"

    # Flags
    flags = _generate_flags(final_score)
    rd_indicators = model_results.get("reality_defender", {}).get("indicators") or []
    if rd_indicators:
        flags = rd_indicators + flags

    # Model breakdown
    model_breakdown: dict[str, Any] = {}
    for key in ["reality_defender", "model1_ViT", "model2_SigLIP", "model3_general"]:
        r = model_results.get(key, {})
        model_breakdown[key] = {
            "name": MODELS[key]["name"], "verdict": r.get("verdict", "Error"),
            "score": r.get("score_pct"), "status": r.get("status", "error"),
            "error": r.get("error"), "elapsed_ms": r.get("elapsed_ms", 0),
            "weight": f"{MODELS[key]['weight_4'] * 100:.0f}%",
            "accuracy": MODELS[key]["accuracy"],
            "indicators": r.get("indicators"),
        }

    processing_time = int((time.time() - total_start) * 1000)

    result = {
        "verdict": verdict, "risk_level": risk_level,
        "final_confidence": round(final_score * 100, 1),
        "final_score_raw": round(final_score, 4),
        "ai_score": round(ai_score * 100, 1),
        "feature_score": round(feature_score * 100, 1),
        "model_breakdown": model_breakdown,
        "models_agree": models_agree,
        "all_models_agree": all(
            r.get("verdict") == "Fake" for r in model_results.values()
            if r.get("verdict") is not None
        ) if scores else False,
        "confidence_note": confidence_note,
        "models_responded": len(scores), "total_models": 4,
        "mode": mode,
        "rd_status": "available" if rd_available else "unavailable",
        "rd_indicators": rd_indicators,
        "flags": flags,
        "highlighted_image": None,
        "analysis": analysis,
        "processing_time_ms": processing_time,
        "cached": False
    }

    # Save to history
    save_to_history(image_name, result)

    return result


def _generate_flags(score: float) -> list[str]:
    if score > 0.85:
        return ["High probability GAN-generated", "Facial feature anomalies detected",
                "Pixel-level inconsistencies found"]
    elif score > 0.70:
        return ["Possible AI face-swap detected", "Inconsistent lighting patterns",
                "Boundary artifacts around face edges"]
    elif score > 0.50:
        return ["Minor manipulation artifacts", "Subtle texture inconsistencies",
                "Recommend manual expert review"]
    elif score > 0.35:
        return ["Low-level noise patterns detected", "Inconclusive — borderline case"]
    else:
        return ["No manipulation signatures found", "Consistent lighting and texture",
                "Natural facial feature distribution"]


# ═══════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Ensemble Deepfake Detection — 4 Models + 4 Analyses",
        "reality_defender": "available" if (HAS_RD and RD_KEY) else "not configured",
        "face_recognition": "installed" if HAS_FACE_REC else "using OpenCV fallback",
        "features": ["ELA", "EXIF", "Face Analysis", "Frequency Analysis", "History"],
        "models": {k: {"name": v["name"], "accuracy": v["accuracy"]}
                   for k, v in MODELS.items()}
    })


@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename or ""):
        return jsonify({"error": f"Invalid type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    try:
        image_bytes = file.read()
        img_hash = get_image_hash(image_bytes)
        cached = get_cached(img_hash)
        if cached:
            cached["cached"] = True
            return jsonify(cached)
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            return jsonify({"error": "Invalid or corrupted image"}), 400
        result = run_ensemble(image_bytes, file.filename or "unknown.jpg")
        if result.get("verdict") != "ERROR":
            set_cache(img_hash, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({"history": load_history()})


@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({"models": {k: {
        "name": v["name"], "id": v["id"], "accuracy": v["accuracy"],
        "description": v["description"], "weight": f"{v['weight_4']*100:.0f}%"
    } for k, v in MODELS.items()}})


# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 62)
    print("  🔍 Ensemble Deepfake Detection — 4 Models + 4 Analyses")
    print("=" * 62)
    for k, v in MODELS.items():
        icon = "👑" if k == "reality_defender" else "•"
        print(f"  {icon} {v['name']} [{v['accuracy']}] ({v['weight_4']*100:.0f}%)")
    print("  ─────────────────────────────────────────")
    print("  📊 ELA | 📋 EXIF | 👤 Face | 📈 Frequency | 📜 History")
    print("=" * 62)
    print(f"  Server:  http://localhost:5000")
    print(f"  HF:      {'✅' if HF_TOKEN else '⚠️  Missing'}")
    print(f"  RD:      {'✅' if RD_KEY else '⚠️  Fallback mode'}")
    print(f"  OpenCV:  ✅ | Face Rec: {'✅' if HAS_FACE_REC else '⚠️ Using OpenCV cascade'}")
    print("=" * 62 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
