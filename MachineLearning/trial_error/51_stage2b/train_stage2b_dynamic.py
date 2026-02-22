import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import joblib
import xgboost as xgb
from transformers import pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
SEED = 43

print("🚀 DYNAMIC TRAINING STAGE 2B")
print("   Strategi: Menangkap 'Fluktuasi Emosi' & 'Warna Musik' (Std Dev + Tonnetz)")

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

# ================= 2. DYNAMIC FEATURE EXTRACTION =================
print("\n⏳ Extracting Advanced Physics (Dynamics & Harmony)...")

def get_dynamic_features(path, lyric):
    try:
        y, sr = librosa.load(path, sr=16000)
        
        # 1. DINAMIKA (Fluctuation)
        # Kita tidak pakai Mean RMS (karena sama), kita pakai STD (Variasi)
        rms = librosa.feature.rms(y=y)
        rms_std = np.std(rms) # Seberapa naik-turun volumenya?
        
        # 2. HARMONI (Musicality)
        # Tonnetz mendeteksi perubahan Tonal (Mayor/Minor/Fifth)
        # Kita ambil rata-rata variasi harmoninya
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonal_var = np.mean(np.std(tonnetz, axis=1)) 
        
        # 3. TEXTURE (Timbre)
        # Spectral Contrast: Perbedaan peak dan valley di frekuensi
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast)
        
        # 4. TEMPO (Tetap berguna)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.ndim(tempo) > 0: tempo = tempo[0]

    except:
        rms_std, tonal_var, contrast_mean, tempo = 0, 0, 0, 0
        
    # --- TEXT (Tetap Kunci) ---
    s_joy, s_sad = 0, 0
    if lyric and str(lyric).strip():
        text = re.sub(r"[^a-z0-9\s]", '', str(lyric).lower())[:512]
        try:
            res = nlp(text)[0]
            s = {item['label']: item['score'] for item in res}
            s_joy = s.get('joy', 0)
            s_sad = s.get('sadness', 0) + s.get('fear', 0)
        except: pass
        
    # FITUR BARU: [Fluktuasi Vol, Variasi Harmoni, Tekstur, Tempo, Joy, Sad]
    return [rms_std, tonal_var, contrast_mean, tempo, s_joy, s_sad]

X = []
y = []

for p in tqdm(raw_files):
    fid = os.path.basename(p).split('_')[0]
    if fid not in mood_map: continue
    label_str = mood_map[fid]
    if label_str not in ['sad', 'relaxed']: continue
    
    # 0=Sad, 1=Relaxed
    lbl = 0 if label_str == 'sad' else 1
    
    feats = get_dynamic_features(p, lyrics_map.get(fid, ""))
    X.append(feats)
    y.append(lbl)

X = np.array(X)
y = np.array(y)

# ================= 3. TRAINING XGBOOST =================
print("\n🔥 Training XGBoost on Dynamic Features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4, 
    subsample=0.8,
    colsample_bytree=0.8, # Random feature per tree (biar gak terpaku ke 1 fitur)
    random_state=SEED
)

# Cross Validation
cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

print("\n" + "="*50)
print(f"🏆 DYNAMIC MODEL RESULT")
print("="*50)
print(f"✅ CV Accuracy : {np.mean(cv_scores)*100:.2f}%")
print(f"📉 Std Dev     : ±{np.std(cv_scores)*100:.2f}%")
print("-" * 50)

clf.fit(X_scaled, y)
print("Training Accuracy:", clf.score(X_scaled, y))

# Save
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(clf, 'models/stage2b_dynamic_xgb.pkl')
joblib.dump(scaler, 'models/stage2b_dynamic_scaler.pkl') # Ganti nama scaler biar gak ketukar

print("\n✅ Model Disimpan: 'models/stage2b_dynamic_xgb.pkl'")