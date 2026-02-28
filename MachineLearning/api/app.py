import os
import re
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

# --- IMPORT UTILS V1 (Wajib upload utils.py) ---
try:
    from utils import extract_features_from_file
except ImportError:
    print("[WARN] Warning: utils.py tidak ditemukan. Endpoint V1 mungkin error.")
    def extract_features_from_file(path): return None

# Mute Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ==============================================================================
# 1. KONFIGURASI
# ==============================================================================
MODEL_REPO_ID = os.environ.get('MODEL_REPO_ID', 'xullfikar/roodio-models')
MODEL_DIR = 'models'
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device Config (Gunakan CPU di Hugging Face Free Tier)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Server running on: {DEVICE}")

# Mapping Database (Global)
MOOD_DB_MAPPING = {
    'happy':   'MD-0000001',
    'sad':     'MD-0000002',
    'relaxed': 'MD-0000003',
    'angry':   'MD-0000004'
}

# ==============================================================================
# 2. DOWNLOAD MODELS DARI HUGGING FACE HUB
# ==============================================================================
_models_exist = os.path.isdir(os.path.join(MODEL_DIR, 'v3'))

if _models_exist:
    print(f"[INFO] Local models/v3 folder found — skipping HuggingFace Hub download.")
else:
    print(f"[INFO] Downloading v3 models from HuggingFace Hub ({MODEL_REPO_ID})...")
    try:
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=MODEL_DIR,
            repo_type="model",
            allow_patterns=["v3/*"]
        )
        print("[OK] V3 Models downloaded successfully from Hub.")
    except Exception as e:
        print(f"[WARN] Hub download failed: {e}")
        print("   Falling back to local models/ directory if available.")

# ==============================================================================
# 3. DEFINISI ARSITEKTUR STAGE 1 (PYTORCH)
# ==============================================================================
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=3074):
        super(AudioClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(256, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)

# ==============================================================================
# 4. LOAD MODELS (V1, V2 & V3)
# ==============================================================================
print("[INFO] Loading Models...")

# --- A. LOAD V1 MODELS (Legacy) ---
try:
    v1_model = joblib.load(os.path.join(MODEL_DIR, 'v1/audio_mood_model.pkl'))
    v1_columns = joblib.load(os.path.join(MODEL_DIR, 'v1/feature_columns.pkl'))
    v1_encoder = joblib.load(os.path.join(MODEL_DIR, 'v1/label_encoder.pkl'))
    print("[OK] V1 Models Loaded.")
except Exception as e:
    print(f"[WARN] V1 Load Error: {e}")
    v1_model = None

# --- B. SHARED: YAMNet (Dipakai V2 & V3) ---
try:
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("[OK] YAMNet Loaded.")
except Exception as e:
    print(f"[ERROR] YAMNet Load Error: {e}")
    yamnet_model = None

# --- C. LOAD V2 MODELS (PyTorch Hybrid) ---
v2_loaded = False
try:
    # Stage 1 (Audio PyTorch)
    s1_v2_path = os.path.join(MODEL_DIR, 'v2/stage1_nn.pth')
    if os.path.exists(s1_v2_path):
        s1_v2_model = AudioClassifier().to(DEVICE)
        s1_v2_model.load_state_dict(torch.load(s1_v2_path, map_location=DEVICE))
        s1_v2_model.eval()
    else:
        raise FileNotFoundError("v2/stage1_nn.pth not found")

    # Encoder Stage 1
    enc_v2_path = os.path.join(MODEL_DIR, 'v2/stage1_encoder.pkl')
    s1_v2_encoder = joblib.load(enc_v2_path) if os.path.exists(enc_v2_path) else None

    # Stage 2A (Random Forest)
    s2a_v2_rf = joblib.load(os.path.join(MODEL_DIR, 'v2/stage2a_rf.pkl'))
    s2a_v2_meta = joblib.load(os.path.join(MODEL_DIR, 'v2/stage2a_meta.pkl'))

    # Stage 2B (BERT)
    bert_v2_path = os.path.join(MODEL_DIR, 'v2/model_hybrid_final')
    tokenizer_v2 = AutoTokenizer.from_pretrained(bert_v2_path)
    bert_v2_model = AutoModelForSequenceClassification.from_pretrained(bert_v2_path).to(DEVICE)
    bert_v2_model.eval()

    v2_loaded = True
    print("[OK] V2 Models Loaded Successfully.")
except Exception as e:
    print(f"[WARN] V2 Load Error: {e}")

# --- D. LOAD V3 MODELS (Latest) ---
v3_loaded = False
try:
    # Stage 1 (Audio PyTorch)
    s1_v3_path = os.path.join(MODEL_DIR, 'v3/stage1_nn.pth')
    if os.path.exists(s1_v3_path):
        s1_v3_model = AudioClassifier().to(DEVICE)
        s1_v3_model.load_state_dict(torch.load(s1_v3_path, map_location=DEVICE))
        s1_v3_model.eval()
    else:
        raise FileNotFoundError("v3/stage1_nn.pth not found")

    # Encoder Stage 1
    enc_v3_path = os.path.join(MODEL_DIR, 'v3/stage1_encoder.pkl')
    s1_v3_encoder = joblib.load(enc_v3_path) if os.path.exists(enc_v3_path) else None

    # Stage 2A (Random Forest)
    s2a_v3_rf = joblib.load(os.path.join(MODEL_DIR, 'v3/stage2a_rf.pkl'))
    s2a_v3_meta = joblib.load(os.path.join(MODEL_DIR, 'v3/stage2a_meta.pkl'))

    # Stage 2B (BERT)
    bert_v3_path = os.path.join(MODEL_DIR, 'v3/model_stage2b')
    tokenizer_v3 = AutoTokenizer.from_pretrained(bert_v3_path)
    bert_v3_model = AutoModelForSequenceClassification.from_pretrained(bert_v3_path).to(DEVICE)
    bert_v3_model.eval()

    v3_loaded = True
    print("[OK] V3 Models Loaded Successfully.")
except Exception as e:
    print(f"[ERROR] V3 Load Error: {e}")

# ==============================================================================
# 5. HELPER FUNCTIONS
# ==============================================================================
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def trim_middle(y, sr=16000, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def extract_features_v2(y, sr):
    """Fitur: YAMNet + RMS + ZCR (3074 Dimensi) — Dipakai V2 & V3"""
    y_trim = trim_middle(y, sr)
    if np.max(np.abs(y_trim)) > 0:
        y_norm = y_trim / np.max(np.abs(y_trim))
    else:
        y_norm = y_trim

    if len(y_norm) < 16000: y_norm = np.pad(y_norm, (0, 16000 - len(y_norm)))

    _, embeddings, _ = yamnet_model(y_norm)

    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_std  = tf.math.reduce_std(embeddings, axis=0).numpy()
    emb_max  = tf.reduce_max(embeddings, axis=0).numpy()

    rms = np.mean(librosa.feature.rms(y=y_trim))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_trim))

    return np.concatenate([emb_mean, emb_std, emb_max, [rms, zcr]])

def extract_rf_features(y, sr):
    """Fitur RF: YAMNet Mean + RMS + ZCR — Dipakai V2 & V3"""
    if len(y) < 16000: y = np.pad(y, (0, 16000 - len(y)))
    _, embeddings, _ = yamnet_model(y)
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([emb_mean, [rms, zcr]])


def run_hybrid_prediction(y, sr, lyrics_text, s1_model, s1_encoder,
                          s2a_rf, s2a_meta, bert_tokenizer, bert_mod):
    """
    Shared prediction logic untuk V2 & V3.
    Karena pipeline-nya identik (Stage1 → Branch High/Low → Stage2A/2B),
    hanya weight model-nya yang berbeda.
    """
    # 1. Stage 1 (PyTorch)
    feat_s1 = extract_features_v2(y, sr)
    feat_tensor = torch.tensor([feat_s1], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = s1_model(feat_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)

    if s1_encoder:
        s1_label = s1_encoder.inverse_transform([pred_idx])[0].lower()
    else:
        s1_label = "high" if pred_idx == 0 else "low"

    is_high = 'high' in s1_label or 'angry' in s1_label or 'happy' in s1_label

    final_mood = ""
    conf = 0.0
    reason = ""

    if is_high:
        # Branch High Energy (RF)
        feat_rf = extract_rf_features(y, sr).reshape(1, -1)
        dummy_txt = np.array([[0.5, 0.5]])
        p_rf = s2a_rf.predict_proba(feat_rf)
        meta_in = np.concatenate([p_rf, dummy_txt], axis=1)

        idx = s2a_meta.predict(meta_in)[0]
        conf = float(s2a_meta.predict_proba(meta_in)[0][idx])
        final_mood = "angry" if idx == 0 else "happy"
        reason = "High Energy -> Audio Analysis"
    else:
        # Branch Low Energy (BERT)
        txt = clean_text(lyrics_text)
        if len(txt) < 5:
            final_mood = "relaxed"
            conf = 0.5
            reason = "Low Energy -> No Lyrics"
        else:
            inputs = bert_tokenizer(txt, return_tensors="pt", truncation=True,
                                    padding=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = bert_mod(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

            p_sad, p_rel = probs[0].item(), probs[1].item()
            if p_sad > p_rel:
                final_mood = "sad"
                conf = p_sad
            else:
                final_mood = "relaxed"
                conf = p_rel
            reason = "Low Energy -> Lyrics Analysis"

    return final_mood, conf, reason, s1_label


# ==============================================================================
# 6. ROUTES
# ==============================================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "system": "Roodio AI API",
        "active_model": "v3",
        "model_source": MODEL_REPO_ID,
        "endpoints": {
            "v1_legacy": "/predict/v1",
            "v2_hybrid": "/predict/v2",
            "v3_latest": "/predict"
        }
    })

# --- ENDPOINT V1 (LEGACY) ---
@app.route('/predict/v1', methods=['POST'])
def predict_v1():
    if not v1_model:
        return jsonify({"status": "error", "message": "Model V1 Failed."}), 500

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, "v1_" + file.filename)

    try:
        file.save(filepath)
        features = extract_features_from_file(filepath)

        if os.path.exists(filepath): os.remove(filepath)
        if not features:
            return jsonify({"status": "error", "message": "Extraction failed."}), 400

        input_df = pd.DataFrame([features]).reindex(columns=v1_columns, fill_value=0)
        pred_idx = v1_model.predict(input_df)[0]

        confidence = 0.0
        if hasattr(v1_model, "predict_proba"):
            confidence = float(max(v1_model.predict_proba(input_df)[0]))

        mood_text = v1_encoder.inverse_transform([pred_idx])[0].lower()

        return jsonify({
            "status": "success",
            "version": "v1_legacy",
            "data": {
                "mood": mood_text,
                "mood_id": MOOD_DB_MAPPING.get(mood_text, "MD-UNKNOWN"),
                "confidence": round(confidence * 100, 2)
            }
        })
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"status": "error", "message": str(e)}), 500

# --- ENDPOINT V2 (HYBRID PYTORCH — ARCHIVED) ---
@app.route('/predict/v2', methods=['POST'])
def predict_v2():
    if not v2_loaded:
        return jsonify({"error": "V2 Models not loaded."}), 500

    if 'file' not in request.files: return jsonify({"error": "No file"}), 400

    file = request.files['file']
    lyrics_text = request.form.get('lyrics', '')
    filepath = os.path.join(UPLOAD_FOLDER, "v2_" + file.filename)
    file.save(filepath)

    try:
        y, sr = librosa.load(filepath, sr=16000)

        final_mood, conf, reason, s1_label = run_hybrid_prediction(
            y, sr, lyrics_text,
            s1_v2_model, s1_v2_encoder,
            s2a_v2_rf, s2a_v2_meta,
            tokenizer_v2, bert_v2_model
        )

        if os.path.exists(filepath): os.remove(filepath)

        return jsonify({
            "status": "success",
            "version": "v2_hybrid",
            "data": {
                "mood": final_mood,
                "mood_id": MOOD_DB_MAPPING.get(final_mood, "MD-UNKNOWN"),
                "confidence": round(conf * 100, 2),
                "reasoning": reason,
                "stage1_raw": s1_label
            }
        })
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"error": str(e)}), 500

# --- ENDPOINT V3 (LATEST — DEFAULT /predict) ---
@app.route('/predict', methods=['POST'])
def predict_v3():
    if not v3_loaded:
        return jsonify({"error": "V3 Models not loaded."}), 500

    if 'file' not in request.files: return jsonify({"error": "No file"}), 400

    file = request.files['file']
    lyrics_text = request.form.get('lyrics', '')
    filepath = os.path.join(UPLOAD_FOLDER, "v3_" + file.filename)
    file.save(filepath)

    try:
        y, sr = librosa.load(filepath, sr=16000)

        final_mood, conf, reason, s1_label = run_hybrid_prediction(
            y, sr, lyrics_text,
            s1_v3_model, s1_v3_encoder,
            s2a_v3_rf, s2a_v3_meta,
            tokenizer_v3, bert_v3_model
        )

        if os.path.exists(filepath): os.remove(filepath)

        return jsonify({
            "status": "success",
            "version": "v3_latest",
            "data": {
                "mood": final_mood,
                "mood_id": MOOD_DB_MAPPING.get(final_mood, "MD-UNKNOWN"),
                "confidence": round(conf * 100, 2),
                "reasoning": reason,
                "stage1_raw": s1_label
            }
        })
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)