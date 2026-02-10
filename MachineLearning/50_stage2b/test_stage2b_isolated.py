import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIGURATION =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
MODEL_PATH = "models/stage2b_purified_rf.pkl"

# FILTER CONFIG (SAMA PERSIS DENGAN SAAT TRAINING)
# Kita hanya akan mengetes data yang lolos filter ini
MAX_SADNESS_FOR_RELAXED = 0.6 
MAX_JOY_FOR_SAD = 0.6

print("🚀 TESTING ON PURIFIED DATA (TRAINING SET ONLY)")
print("   Menguji model HANYA pada 80 file bersih yang dipakai untuk belajar.")

# ================= 1. PREPARE RESOURCES =================
try:
    print("\n⏳ Loading Lyrics...")
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    mood_map = dict(zip(df['id'], df['mood']))
    
    print("⏳ Loading Models...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    nlp_filter = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)
    model_rf = joblib.load(MODEL_PATH)
    
    print("✅ System Ready.")

except Exception as e:
    print(f"❌ Error Setup: {e}")
    exit()

# ================= 2. FEATURE EXTRACTOR =================
def get_features(path, lyric):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
        
        _, emb, _ = yamnet_model(y)
        aud_vec = tf.reduce_mean(emb, axis=0).numpy()
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
        rms = np.mean(librosa.feature.rms(y=y))
        feat_aud = np.concatenate([aud_vec, chroma, [rms]])
        
        text = re.sub(r"[^a-z0-9\s]", '', str(lyric).lower())[:512]
        s_sad, s_rel = 0, 0
        if text:
            try:
                res = nlp_filter(text)[0] 
                s = {item['label']: item['score'] for item in res}
                s_sad = s.get('sadness', 0) + s.get('fear', 0)
                s_rel = s.get('joy', 0) + s.get('neutral', 0)
            except: pass
        feat_txt = [s_sad, s_rel]
        
        return np.concatenate([feat_aud, feat_txt])
    except: return None

# Helper Filter Sentiment
def get_sentiment_score(text):
    if not text or str(text).strip() == "": return {'sadness': 0, 'joy': 0}
    text = re.sub(r"[^a-z0-9\s]", '', str(text).lower())[:512]
    try:
        res = nlp_filter(text)[0]
        return {item['label']: item['score'] for item in res}
    except: return {'sadness': 0, 'joy': 0}

# ================= 3. TESTING LOOP =================
y_true = []
y_pred = []
accepted_files = []

# Kumpulkan semua file dulu
all_files = []
for d in SOURCE_DIRS:
    for m in TARGET_MOODS:
        all_files.extend(glob.glob(os.path.join(d, m, '*.wav')) + glob.glob(os.path.join(d, m, '*.mp3')))

print(f"\n📂 Scanning & Filtering {len(all_files)} files...")

for path in tqdm(all_files):
    fid = os.path.basename(path).split('_')[0]
    
    if fid not in mood_map: continue
    label_str = mood_map[fid]
    lyric = lyrics_map.get(fid, "")
    
    # --- LOGIKA FILTER (SAMA DENGAN TRAINING) ---
    scores = get_sentiment_score(lyric)
    keep = True
    
    if label_str == 'relaxed':
        # Kalau Relaxed tapi lirik sedih -> BUANG (Jangan Dites)
        if scores['sadness'] > MAX_SADNESS_FOR_RELAXED: keep = False
    elif label_str == 'sad':
        # Kalau Sad tapi lirik happy -> BUANG (Jangan Dites)
        if scores['joy'] > MAX_JOY_FOR_SAD: keep = False
    
    # KITA HANYA TES JIKA 'keep == True' (Data Bersih)
    if not keep:
        continue 
        
    # --- PROSES TESTING ---
    # 0=Sad, 1=Relaxed
    true_label = 0 if label_str == 'sad' else 1
    
    feat = get_features(path, lyric)
    if feat is not None:
        feat = feat.reshape(1, -1)
        pred_label = model_rf.predict(feat)[0]
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        accepted_files.append(os.path.basename(path))

# ================= 4. REPORTING =================
print("\n" + "="*60)
print("📊 REPORT: PURIFIED DATA PERFORMANCE")
print("="*60)
print(f"Total Data Terpilih : {len(y_true)} (Sesuai dengan jumlah 'Total Latih' tadi)")
print("-" * 60)

acc = accuracy_score(y_true, y_pred) * 100
print(f"🏆 TRAINING SET ACCURACY: {acc:.2f}%")
print("-" * 60)
print(classification_report(y_true, y_pred, target_names=['Sad', 'Relaxed']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Sad', 'Relaxed'], yticklabels=['Sad', 'Relaxed'])
plt.title(f'Confusion Matrix (Training/Purified Data)\nAcc: {acc:.2f}%')
plt.xlabel('Prediksi Model')
plt.ylabel('Label Asli')
plt.tight_layout()
plt.savefig('stage2b_purified_cm.png')
plt.show()

print("✅ Selesai. Cek 'stage2b_purified_cm.png'.")