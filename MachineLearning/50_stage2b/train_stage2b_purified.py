import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
SEED = 43

# THRESHOLD "KEJUJURAN DATA"
# Lagu Relaxed dilarang sedih!
MAX_SADNESS_FOR_RELAXED = 0.6 
# Lagu Sad dilarang happy!
MAX_JOY_FOR_SAD = 0.6

print("🚀 TRAINING STAGE 2B: PURIFIED DATASET")
print("   Strategy: Remove 'Confusing' songs (e.g., Relaxed songs with Sad lyrics)")

# ================= 1. LOAD LYRICS & MOOD =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(['sad', 'relaxed'])]
    
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    mood_map = dict(zip(df['id'], df['mood']))
    for k, v in lyrics_map.items():
        if pd.isna(v): lyrics_map[k] = ""
            
    print(f"📊 Excel Database: {len(df)} entries.")
except Exception as e:
    print(f"❌ Error Excel: {e}")
    exit()

# ================= 2. ANALYZE & PURIFY DATA =================
print("\n🧹 Purifying Data based on Lyrics Sentiment...")

nlp = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

def get_sentiment(text):
    if not text or str(text).strip() == "": return {'sadness': 0, 'joy': 0}
    text = re.sub(r"[^a-z0-9\s]", '', str(text).lower())[:512]
    try:
        res = nlp(text)[0]
        return {item['label']: item['score'] for item in res}
    except: return {'sadness': 0, 'joy': 0}

valid_files = []
valid_labels = [] # 0=Sad, 1=Relaxed
dropped_count = 0

raw_files = []
for d in SOURCE_DIRS:
    raw_files.extend(glob.glob(os.path.join(d, 'sad', '*.wav')) + glob.glob(os.path.join(d, 'sad', '*.mp3')))
    raw_files.extend(glob.glob(os.path.join(d, 'relaxed', '*.wav')) + glob.glob(os.path.join(d, 'relaxed', '*.mp3')))

for p in tqdm(raw_files, desc="Filtering"):
    base = os.path.basename(p)
    fid = base.split('_')[0] if '_' in base else None
    
    if fid not in mood_map: continue
    
    label_str = mood_map[fid]
    lyric = lyrics_map[fid]
    
    # Cek Sentimen Lirik
    scores = get_sentiment(lyric)
    
    # --- LOGIKA PURIFIKASI ---
    keep = True
    
    if label_str == 'relaxed':
        # Jika Relaxed tapi liriknya SANGAT SEDIH -> BUANG
        if scores['sadness'] > MAX_SADNESS_FOR_RELAXED:
            keep = False
        else:
            lbl = 1
            
    elif label_str == 'sad':
        # Jika Sad tapi liriknya SANGAT HAPPY -> BUANG
        if scores['joy'] > MAX_JOY_FOR_SAD:
            keep = False
        else:
            lbl = 0
            
    if keep:
        valid_files.append(p)
        valid_labels.append(lbl)
    else:
        dropped_count += 1

print(f"\n✅ DATASET BERSIH TERBENTUK!")
print(f"   Total Awal   : {len(raw_files)}")
print(f"   Dibuang      : {dropped_count} (Data Ambigu/Confusing)")
print(f"   Total Latih  : {len(valid_files)}")
print(f"   Label 0 (Sad): {valid_labels.count(0)}")
print(f"   Label 1 (Rel): {valid_labels.count(1)}")

# ================= 3. FEATURE EXTRACTION (AUDIO ONLY IS ENOUGH NOW) =================
# Karena data sudah 'bersih' secara lirik, audio statis pun harusnya lebih konsisten
# Tapi kita tetap pakai Stacking Audio+Text biar aman

print("\n⏳ Extracting Features for Training...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def get_features(path, lyric):
    # 1. Audio
    y, sr = librosa.load(path, sr=16000)
    if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
    _, emb, _ = yamnet(y)
    aud_vec = tf.reduce_mean(emb, axis=0).numpy()
    chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    feat_aud = np.concatenate([aud_vec, chroma, [rms]])
    
    # 2. Text (Tetap dipakai sebagai fitur)
    scores = get_sentiment(lyric)
    # Mapping simple: Sadness vs (Joy + Neutral)
    s_sad = scores.get('sadness', 0) + scores.get('fear', 0)
    s_rel = scores.get('joy', 0) + scores.get('neutral', 0)
    feat_txt = [s_sad, s_rel]
    
    return np.concatenate([feat_aud, feat_txt])

X = []
for i in tqdm(range(len(valid_files)), desc="Extraction"):
    p = valid_files[i]
    fid = os.path.basename(p).split('_')[0]
    lyr = lyrics_map.get(fid, "")
    X.append(get_features(p, lyr))

X = np.array(X)
y = np.array(valid_labels)

# ================= 4. TRAIN RANDOM FOREST =================
print("\n🔥 Training Robust Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced', # Penting jika jumlah data tidak seimbang setelah filtering
    random_state=SEED,
    n_jobs=-1
)

# Cross Validation Check
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')

print("\n" + "="*50)
print(f"🏆 PURIFIED MODEL PERFORMANCE")
print("="*50)
print(f"✅ Mean Accuracy : {np.mean(scores)*100:.2f}%")
print(f"📉 Std Dev       : ±{np.std(scores)*100:.2f}%")

# Fit & Save
rf.fit(X, y)

# Save as SINGLE model (Lebih simpel, RF handle audio+text sekaligus)
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(rf, 'models/stage2b_purified_rf.pkl')

with open('models/stage2b_info.txt', 'w') as f:
    f.write(f"Model: Random Forest on Purified Data (Audio+Text)\n")
    f.write(f"Filter Logic: Relaxed cannot have Sadness > {MAX_SADNESS_FOR_RELAXED}\n")
    f.write(f"Accuracy: {np.mean(scores)*100:.2f}%\n")

print("✅ Model Saved: 'models/stage2b_purified_rf.pkl'")