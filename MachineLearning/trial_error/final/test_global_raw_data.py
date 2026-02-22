import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import joblib 
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

print("🚀 GLOBAL SYSTEM TEST: RAW DATA EVALUATION")
print("   Menguji seluruh pipeline (Stage 1 -> Stage 2A/2B Trifecta) pada data mentah.")
print("="*60)

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
MODEL_DIR = 'models/'

# Target Label yang akan dites
TARGET_CLASSES = ['angry', 'happy', 'sad', 'relaxed']

# ================= 1. LOAD MODELS =================
print("⏳ Loading Models & Resources...")
try:
    # Resources
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)
    
    # Stage 1 (NN)
    s1_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'stage1_nn.h5'))
    
    # Stage 2A (Legacy RF)
    s2a_rf = joblib.load(os.path.join(MODEL_DIR, 'stage2a_rf.pkl'))
    s2a_meta = joblib.load(os.path.join(MODEL_DIR, 'stage2a_meta.pkl'))
    
    # Stage 2B (NEW TRIFECTA XGBOOST)
    s2b_model = joblib.load(os.path.join(MODEL_DIR, 'stage2b_trifecta_xgb.pkl'))
    s2b_scaler = joblib.load(os.path.join(MODEL_DIR, 'stage2b_trifecta_scaler.pkl'))
    
    # Load Lyrics Map
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    
    print("✅ System Ready.")

except Exception as e:
    print(f"❌ Error Loading: {e}")
    exit()

# ================= 2. EXTRACTION FUNCTIONS =================

def trim_middle(y, sr=16000, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# --- S1 ---
def extract_feat_s1(y, sr):
    y = trim_middle(y, sr)
    if np.max(np.abs(y)) > 0: y_norm = y / np.max(np.abs(y))
    else: y_norm = y
    if len(y_norm) < 16000: y_norm = np.pad(y_norm, (0, 16000 - len(y_norm)))
    _, embeddings, _ = yamnet_model(y_norm)
    yamnet_emb = tf.concat([
        tf.reduce_mean(embeddings, axis=0),
        tf.math.reduce_std(embeddings, axis=0),
        tf.reduce_max(embeddings, axis=0)
    ], axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([yamnet_emb, [rms, zcr]]).reshape(1, -1)

# --- S2A (Legacy) ---
def extract_feat_s2a(y, sr, lyrics):
    # Audio
    if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
    _, emb, _ = yamnet_model(y)
    vec = tf.reduce_mean(emb, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    feat_aud = np.concatenate([vec, [rms, zcr]]).reshape(1, -1)
    
    # Text
    chunk = str(lyrics)[:512]
    try:
        out = nlp_classifier(chunk)[0]
        s = {item['label']: item['score'] for item in out}
        val0 = s.get('anger',0) + s.get('disgust',0) + s.get('fear',0) + s.get('sadness', 0)
        val1 = s.get('joy',0) + s.get('surprise',0)
        feat_txt = np.array([[val0, val1]])
    except: 
        feat_txt = np.array([[0.5, 0.5]])
        
    return feat_aud, feat_txt

# --- S2B (TRIFECTA) ---
def extract_feat_s2b_trifecta(y, sr, lyrics):
    # 1. Audio: Contrast
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast)
    except: contrast_mean = 0

    # 2. Text: Sad & Joy
    s_sad, s_joy = 0, 0
    if lyrics and str(lyrics).strip():
        text = re.sub(r"[^a-z0-9\s]", '', str(lyrics).lower())[:512]
        try:
            res = nlp_classifier(text)[0] 
            s = {item['label']: item['score'] for item in res}
            s_sad = s.get('sadness', 0) + s.get('fear', 0)
            s_joy = s.get('joy', 0) + s.get('neutral', 0)
        except: pass
        
    # Return 2D Array untuk Scaler
    return np.array([[contrast_mean, s_sad, s_joy]])

# ================= 3. TESTING LOOP =================

y_true = []
y_pred = []

# Collect Files
all_files = []
for d in SOURCE_DIRS:
    for mood in TARGET_CLASSES:
        files = glob.glob(os.path.join(d, mood, '*.wav')) + glob.glob(os.path.join(d, mood, '*.mp3'))
        for f in files:
            all_files.append((f, mood))

print(f"\n📂 Scanning {len(all_files)} files total...")

for path, true_label in tqdm(all_files):
    try:
        # Load Audio
        y, sr = librosa.load(path, sr=16000)
        fid = os.path.basename(path).split('_')[0]
        lyrics = lyrics_map.get(fid, "")
        
        # === PREDICTION PIPELINE ===
        predicted_mood = "unknown"
        
        # 1. STAGE 1 Check
        f1 = extract_feat_s1(y, sr)
        p1 = s1_model.predict(f1, verbose=0)[0]
        
        # Asumsi Training Stage 1: 0=High Energy, 1=Low Energy
        # (Jika terbalik, ubah logika if di bawah)
        if p1[0] > p1[1]: 
            # ---> HIGH ENERGY BRANCH (Angry/Happy)
            f_aud, f_txt = extract_feat_s2a(y, sr, lyrics)
            p_rf = s2a_rf.predict_proba(f_aud)
            meta_in = np.concatenate([p_rf, f_txt], axis=1)
            idx = s2a_meta.predict(meta_in)[0]
            
            predicted_mood = "angry" if idx == 0 else "happy"
            
        else:
            # ---> LOW ENERGY BRANCH (Sad/Relaxed) - TRIFECTA
            raw_feats = extract_feat_s2b_trifecta(y, sr, lyrics)
            scaled_feats = s2b_scaler.transform(raw_feats)
            probs = s2b_model.predict_proba(scaled_feats)[0]
            
            # 0=Sad, 1=Relaxed
            if probs[0] > probs[1]:
                predicted_mood = "sad"
            else:
                predicted_mood = "relaxed"
                
        # Record
        y_true.append(true_label)
        y_pred.append(predicted_mood)
        
    except Exception as e:
        # Skip file corrupt
        continue

# ================= 4. FINAL REPORT =================
print("\n" + "="*60)
print("📊 GLOBAL ACCURACY REPORT")
print("="*60)

acc = accuracy_score(y_true, y_pred) * 100
print(f"🏆 TOTAL SYSTEM ACCURACY: {acc:.2f}%")
print("-" * 60)
print(classification_report(y_true, y_pred, target_names=TARGET_CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
plt.title(f'Global Confusion Matrix\nAccuracy: {acc:.2f}%')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('global_test_results.png')
plt.show()

print("\n✅ Selesai. Grafik disimpan ke 'global_test_results.png'")