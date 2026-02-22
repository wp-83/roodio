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

print("🚀 GLOBAL SYSTEM TEST: FINAL CLEAN DATASET")
print("   Menguji pipeline lengkap pada folder 'data/FINAL_DATASET_CLEAN'.")
print("="*60)

# ================= CONFIG =================
# Folder Dataset Baru (Hasil Merger tadi)
TEST_DIR = 'data/FINAL_DATASET_CLEAN' 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
MODEL_DIR = 'models/'

# Target Label
TARGET_CLASSES = ['angry', 'happy', 'sad', 'relaxed']

# ================= 1. LOAD MODELS =================
print("⏳ Loading Models & Resources...")
try:
    # 1. Base Resources
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)
    
    # 2. Stage 1 (NN - High/Low)
    s1_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'stage1_nn.h5'))
    
    # 3. Stage 2A (Legacy RF - Angry/Happy)
    s2a_rf = joblib.load(os.path.join(MODEL_DIR, 'stage2a_rf.pkl'))
    s2a_meta = joblib.load(os.path.join(MODEL_DIR, 'stage2a_meta.pkl'))
    
    # 4. Stage 2B (NEW TRIFECTA XGBOOST - Sad/Relaxed)
    s2b_model = joblib.load(os.path.join(MODEL_DIR, 'stage2b_trifecta_xgb.pkl'))
    s2b_scaler = joblib.load(os.path.join(MODEL_DIR, 'stage2b_trifecta_scaler.pkl'))
    
    # 5. Lyrics
    df_lyric = pd.read_excel(LYRICS_PATH)
    df_lyric['id'] = df_lyric['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    lyrics_map = dict(zip(df_lyric['id'], df_lyric['lyrics']))
    
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

# --- S1 (Yamnet + RMS + ZCR) ---
def extract_feat_s1(y, sr):
    y_trim = trim_middle(y, sr)
    if np.max(np.abs(y_trim)) > 0: y_norm = y_trim / np.max(np.abs(y_trim))
    else: y_norm = y_trim
    if len(y_norm) < 16000: y_norm = np.pad(y_norm, (0, 16000 - len(y_norm)))
    
    _, embeddings, _ = yamnet_model(y_norm)
    yamnet_emb = tf.concat([
        tf.reduce_mean(embeddings, axis=0),
        tf.math.reduce_std(embeddings, axis=0),
        tf.reduce_max(embeddings, axis=0)
    ], axis=0).numpy()
    
    rms = np.mean(librosa.feature.rms(y=y_trim))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_trim))
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

# --- S2B (TRIFECTA - Extract on the fly) ---
# Kita ekstrak langsung dari file audio, BUKAN dari CSV.
# Ini untuk membuktikan bahwa model bisa jalan di dunia nyata.
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
        
    return np.array([[contrast_mean, s_sad, s_joy]])

# ================= 3. TESTING LOOP =================

y_true = []
y_pred = []
files_processed = 0

# Collect Files from CLEAN Directory
all_files = []
if not os.path.exists(TEST_DIR):
    print(f"❌ Folder '{TEST_DIR}' tidak ditemukan! Jalankan script merger dulu.")
    exit()

for mood in TARGET_CLASSES:
    # Cari di subfolder angry, happy, sad, relaxed
    files = glob.glob(os.path.join(TEST_DIR, mood, '*.wav')) + \
            glob.glob(os.path.join(TEST_DIR, mood, '*.mp3'))
    for f in files:
        all_files.append((f, mood))

print(f"\n📂 Processing {len(all_files)} Clean Files...")

for path, true_label in tqdm(all_files):
    try:
        # Load Audio
        y, sr = librosa.load(path, sr=16000)
        fid = os.path.basename(path).split('_')[0]
        lyrics = lyrics_map.get(fid, "")
        
        # === PIPELINE ===
        
        # 1. STAGE 1 (High vs Low)
        feat_s1 = extract_feat_s1(y, sr)
        p1 = s1_model.predict(feat_s1, verbose=0)[0]
        
        # Asumsi Training: 0 = High, 1 = Low
        if p1[0] > p1[1]: 
            # ---> HIGH BRANCH (ANGRY/HAPPY)
            f_aud, f_txt = extract_feat_s2a(y, sr, lyrics)
            p_rf = s2a_rf.predict_proba(f_aud)
            meta_in = np.concatenate([p_rf, f_txt], axis=1)
            idx = s2a_meta.predict(meta_in)[0]
            
            final_pred = "angry" if idx == 0 else "happy"
            
        else:
            # ---> LOW BRANCH (SAD/RELAXED) - TRIFECTA
            # Ekstrak Contrast + Sad + Joy
            raw_feats = extract_feat_s2b_trifecta(y, sr, lyrics)
            
            # Scale
            scaled_feats = s2b_scaler.transform(raw_feats)
            
            # Predict XGBoost
            probs = s2b_model.predict_proba(scaled_feats)[0]
            
            # 0=Sad, 1=Relaxed
            if probs[0] > probs[1]:
                final_pred = "sad"
            else:
                final_pred = "relaxed"
        
        y_true.append(true_label)
        y_pred.append(final_pred)
        files_processed += 1
        
    except Exception as e:
        print(f"Error on {os.path.basename(path)}: {e}")
        continue

# ================= 4. FINAL REPORT =================
print("\n" + "="*60)
print("📊 FINAL GLOBAL ACCURACY REPORT")
print(f"   Dataset: {TEST_DIR}")
print("="*60)

acc = accuracy_score(y_true, y_pred) * 100
print(f"🏆 SYSTEM ACCURACY: {acc:.2f}%")
print("-" * 60)
print(classification_report(y_true, y_pred, target_names=TARGET_CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
plt.title(f'Global System Performance\nClean Dataset (Acc: {acc:.2f}%)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('final_system_performance.png')
plt.show()

print("\n✅ Grafik disimpan ke 'final_system_performance.png'")