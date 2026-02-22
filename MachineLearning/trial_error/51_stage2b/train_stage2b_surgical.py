import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import joblib
import xgboost as xgb  # Kita ganti ke XGBoost
from transformers import pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
SEED = 43

print("🚀 SURGICAL TRAINING STAGE 2B")
print("   Strategi: Hanya pakai fitur 'RMS, Tempo, Joy' (Low Overlap Features)")

# ================= 1. LOAD DATA =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    mood_map = dict(zip(df['id'], df['mood']))
except: exit()

nlp = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

raw_files = []
for d in SOURCE_DIRS:
    raw_files.extend(glob.glob(os.path.join(d, 'sad', '*.wav')) + glob.glob(os.path.join(d, 'sad', '*.mp3')))
    raw_files.extend(glob.glob(os.path.join(d, 'relaxed', '*.wav')) + glob.glob(os.path.join(d, 'relaxed', '*.mp3')))

# ================= 2. SURGICAL FEATURE EXTRACTION =================
# Kita TIDAK PAKAI YAMNet lagi karena terbukti overlap parah di visualisasi
print("\n⏳ Extracting ONLY Key Features (RMS, Tempo, Joy, Sadness)...")

def get_surgical_features(path, lyric):
    # --- AUDIO: Focus on RMS & Tempo (Overlap terendah) ---
    try:
        y, sr = librosa.load(path, sr=16000)
        rms = np.mean(librosa.feature.rms(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.ndim(tempo) > 0: tempo = tempo[0] # Fix format array
    except:
        rms, tempo = 0, 0
        
    # --- TEXT: Focus on Joy & Sadness ---
    s_joy, s_sad = 0, 0
    if lyric and str(lyric).strip():
        text = re.sub(r"[^a-z0-9\s]", '', str(lyric).lower())[:512]
        try:
            res = nlp(text)[0]
            s = {item['label']: item['score'] for item in res}
            s_joy = s.get('joy', 0)
            s_sad = s.get('sadness', 0) + s.get('fear', 0) # Fear sering overlap ke sad
        except: pass
        
    # FITUR KITA CUMA 4 INI! (Kecil tapi Cabe Rawit)
    return [rms, tempo, s_joy, s_sad]

X = []
y = []
filenames = []

for p in tqdm(raw_files):
    fid = os.path.basename(p).split('_')[0]
    if fid not in mood_map: continue
    label_str = mood_map[fid]
    if label_str not in ['sad', 'relaxed']: continue
    
    # 0=Sad, 1=Relaxed
    lbl = 0 if label_str == 'sad' else 1
    
    feats = get_surgical_features(p, lyrics_map.get(fid, ""))
    
    X.append(feats)
    y.append(lbl)
    filenames.append(os.path.basename(p))

X = np.array(X)
y = np.array(y)

# ================= 3. TRAINING XGBOOST =================
print("\n🔥 Training XGBoost (Gradient Boosting)...")

# Scaling penting karena RMS (0.01) dan Tempo (120) beda skala jauh
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model: XGBoost (Lebih jago handle data non-linear dibanding RF)
clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05, # Belajar pelan-pelan
    max_depth=3,        # Pohon pendek (biar gak overfit/menghafal)
    subsample=0.8,      # Pakai sebagian data acak
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=SEED
)

# Cross Validation Jujur
cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

print("\n" + "="*50)
print(f"🏆 SURGICAL MODEL RESULT")
print("="*50)
print(f"✅ CV Accuracy : {np.mean(cv_scores)*100:.2f}%")
print(f"📉 Std Dev     : ±{np.std(cv_scores)*100:.2f}%")
print("-" * 50)

# Train Final
clf.fit(X_scaled, y)
y_pred = clf.predict(X_scaled)
print("Training Accuracy (Memorization Check):")
print(accuracy_score(y, y_pred))

# ================= 4. SAVE =================
# Kita simpan scaler juga karena wajib dipakai saat inference
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(clf, 'models/stage2b_surgical_xgb.pkl')
joblib.dump(scaler, 'models/stage2b_scaler.pkl')

print("\n✅ Model Disimpan: 'models/stage2b_surgical_xgb.pkl'")
print("⚠️ PENTING: Saat inference nanti, fitur input harus di-scale pakai scaler yang disimpan!")