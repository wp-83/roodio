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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
SEED = 43

# Filter Lirik
MAX_SADNESS_FOR_RELAXED = 0.7 
MAX_JOY_FOR_SAD = 0.7

print("🚀 MISSION: BREAKING 80% BARRIER (STAGE 2B) - FIXED")
print("   Strategy: Deep Feature Fusion + Gradient Boosting")

# ================= 1. PREPARE DATA =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    mood_map = dict(zip(df['id'], df['mood']))
    for k, v in lyrics_map.items():
        if pd.isna(v): lyrics_map[k] = ""
except: exit()

# Collect Files
raw_files = []
for d in SOURCE_DIRS:
    raw_files.extend(glob.glob(os.path.join(d, 'sad', '*.wav')) + glob.glob(os.path.join(d, 'sad', '*.mp3')))
    raw_files.extend(glob.glob(os.path.join(d, 'relaxed', '*.wav')) + glob.glob(os.path.join(d, 'relaxed', '*.mp3')))
raw_files.sort()

# --- FILTERING DATA ---
print("\n🧹 Cleaning Data (Removing Conflicting Lyrics)...")
nlp = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

valid_files = []
valid_labels = []
valid_lyrics = []

for p in tqdm(raw_files):
    base = os.path.basename(p)
    fid = base.split('_')[0] if '_' in base else None
    
    if fid not in mood_map: continue
    label_str = mood_map[fid]
    if label_str not in TARGET_MOODS: continue
    
    # Cek Lirik
    lyric = lyrics_map.get(fid, "")
    text = re.sub(r"[^a-z0-9\s]", '', str(lyric).lower())[:512]
    
    keep = True
    scores = {'sadness': 0, 'joy': 0}
    if text:
        try:
            res = nlp(text)[0]
            scores = {item['label']: item['score'] for item in res}
        except: pass
    
    if label_str == 'relaxed':
        if scores['sadness'] > MAX_SADNESS_FOR_RELAXED: keep = False
        lbl = 1
    else: # sad
        if scores['joy'] > MAX_JOY_FOR_SAD: keep = False
        lbl = 0
        
    if keep:
        valid_files.append(p)
        valid_labels.append(lbl)
        valid_lyrics.append(lyric)

print(f"✅ Clean Dataset: {len(valid_files)} files (Sad={valid_labels.count(0)}, Relaxed={valid_labels.count(1)})")

# ================= 2. SUPER FEATURE EXTRACTION =================
print("\n⏳ Extracting FUSION FEATURES (Audio + Text)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def get_super_features(path, lyric):
    # --- A. AUDIO FEATURES ---
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
        
        # 1. YAMNet (1024 dim)
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # 2. Advanced Audio Stats
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spec_cont = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)) 
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1) # 12 dim
        
        # --- FIX ERROR TEMPO DISINI ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.ndim(tempo) > 0: # Jika tempo adalah array, ambil elemen pertamanya
            tempo = tempo[0]
            
        # Gabungkan semua fitur audio manual
        manual_feats = np.array([rms, spec_cent, spec_bw, spec_cont, tempo])
        
        audio_vec = np.concatenate([
            yamnet_vec, 
            chroma, 
            manual_feats
        ])
        
        # --- B. TEXT FEATURES (FULL VECTOR) ---
        emo_vec = [0]*7 
        if lyric and str(lyric).strip():
            text = re.sub(r"[^a-z0-9\s]", '', str(lyric).lower())[:512]
            try:
                res = nlp(text)[0]
                s = {item['label']: item['score'] for item in res}
                emo_vec = [
                    s.get('anger', 0), s.get('disgust', 0), s.get('fear', 0),
                    s.get('joy', 0), s.get('neutral', 0), s.get('sadness', 0),
                    s.get('surprise', 0)
                ]
            except: pass
        
        # --- C. FUSION ---
        return np.concatenate([audio_vec, emo_vec])
        
    except Exception as e:
        print(f"Error extracting {path}: {e}")
        # Return zero vector if error (safeguard)
        return np.zeros(1024 + 12 + 5 + 7)

X = []
for i in tqdm(range(len(valid_files))):
    X.append(get_super_features(valid_files[i], valid_lyrics[i]))

X = np.array(X)
y = np.array(valid_labels)

# ================= 3. TRAINING GRADIENT BOOSTING =================
print("\n🔥 Training Gradient Boosting Classifier...")

# Pipeline: Scale -> Select Best Features -> Boost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(GradientBoostingClassifier(n_estimators=100, random_state=SEED))),
    ('clf', GradientBoostingClassifier(
        n_estimators=300,    
        learning_rate=0.05,  # Learning rate lebih kecil = lebih teliti
        max_depth=4,         
        subsample=0.8,       # Stochastic Gradient Boosting (cegah overfitting)
        random_state=SEED
    ))
])

# Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')

mean_acc = np.mean(scores) * 100
std_acc = np.std(scores) * 100

print("\n" + "="*50)
print(f"🏆 FUSION MODEL RESULTS")
print("="*50)
print(f"✅ Accuracy : {mean_acc:.2f}%")
print(f"📉 Std Dev  : ±{std_acc:.2f}%")

if mean_acc > 80:
    print("🚀 TARGET TERCAPAI! (>80%)")
else:
    print("⚠️ Masih di bawah 80%.")

# ================= 4. SAVE MODEL =================
print("\n💾 Saving Final Fusion Model...")
pipeline.fit(X, y) # Train on full data

if not os.path.exists('models'): os.makedirs('models')
joblib.dump(pipeline, 'models/stage2b_fusion_gb.pkl')

with open('models/stage2b_info.txt', 'w') as f:
    f.write(f"Model: Fusion Gradient Boosting (Audio+Text)\n")
    f.write(f"Accuracy: {mean_acc:.2f}%\n")
    f.write(f"Dataset Size: {len(X)}\n")

print("✅ Model Saved: 'models/stage2b_fusion_gb.pkl'")