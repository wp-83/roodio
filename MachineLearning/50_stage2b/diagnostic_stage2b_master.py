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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
SEED = 42

print("🚀 DIAGNOSTIC MASTER: STAGE 2B (SAD vs RELAXED)")
print("   Tujuan: Menemukan 'Penyebab Kematian' Akurasi.")

# ================= 1. LOAD DATA =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    mood_map = dict(zip(df['id'], df['mood']))
    for k, v in lyrics_map.items(): 
        if pd.isna(v): lyrics_map[k] = ""
except: exit()

files = []
labels = [] 
lyrics_list = []
filenames = []

print("\n🔍 Scanning Data...")
for d in SOURCE_DIRS:
    for m in ['sad', 'relaxed']:
        fs = glob.glob(os.path.join(d, m, '*.wav')) + glob.glob(os.path.join(d, m, '*.mp3'))
        for f in fs:
            fid = os.path.basename(f).split('_')[0]
            if fid in lyrics_map:
                files.append(f)
                filenames.append(os.path.basename(f))
                labels.append(0 if m == 'sad' else 1)
                lyrics_list.append(lyrics_map[fid])

print(f"📦 Total Data: {len(files)} (Sad={labels.count(0)}, Relaxed={labels.count(1)})")

# ================= 2. FEATURE EXTRACTION (GRANULAR) =================
print("\n⏳ Extracting Granular Features (Pisah Audio & Teks)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

# Container Fitur
F_audio_deep = [] # YAMNet
F_audio_stat = [] # RMS, Chroma, Tempo, Centroid
F_text_emo   = [] # 7 Emosi RoBERTa

for i in tqdm(range(len(files))):
    path = files[i]
    lyric = lyrics_list[i]
    
    # --- AUDIO ---
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
        
        # YAMNet
        _, emb, _ = yamnet_model(y)
        F_audio_deep.append(tf.reduce_mean(emb, axis=0).numpy())
        
        # Stats
        rms = np.mean(librosa.feature.rms(y=y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        cont = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.ndim(tempo) > 0: tempo = tempo[0]
        
        F_audio_stat.append([rms, cent, cont, tempo])
        
    except: 
        F_audio_deep.append(np.zeros(1024))
        F_audio_stat.append([0,0,0,0])

    # --- TEXT ---
    emo_vec = [0]*7
    if lyric and str(lyric).strip():
        text = re.sub(r"[^a-z0-9\s]", '', str(lyric).lower())[:512]
        try:
            res = nlp(text)[0]
            s = {item['label']: item['score'] for item in res}
            # anger, disgust, fear, joy, neutral, sadness, surprise
            emo_vec = [
                s.get('anger', 0), s.get('disgust', 0), s.get('fear', 0),
                s.get('joy', 0), s.get('neutral', 0), s.get('sadness', 0),
                s.get('surprise', 0)
            ]
        except: pass
    F_text_emo.append(emo_vec)

# Convert to Numpy
X_deep = np.array(F_audio_deep)
X_stat = np.array(F_audio_stat)
X_text = np.array(F_text_emo)
y = np.array(labels)

# ================= 3. ABLATION STUDY (TESTING PER KOMPONEN) =================
print("\n" + "="*50)
print("🧪 ABLATION STUDY: Fitur Mana yang Berguna?")
print("="*50)

def test_feature(X_in, name):
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    scores = cross_val_score(clf, X_in, y, cv=5)
    print(f"👉 {name:<20} : {np.mean(scores)*100:.2f}% (±{np.std(scores)*100:.2f}%)")
    return np.mean(scores)

# Test 1: Audio Only (Deep)
acc_deep = test_feature(X_deep, "YAMNet (Deep Audio)")

# Test 2: Audio Only (Stats - Tempo/Volume)
acc_stat = test_feature(X_stat, "Stats (RMS/Tempo)")

# Test 3: Text Only (Lyrics)
acc_text = test_feature(X_text, "Lirik (RoBERTa)")

# Test 4: Combined
X_all = np.concatenate([X_deep, X_stat, X_text], axis=1)
acc_all = test_feature(X_all, "COMBINED (All)")

# ================= 4. FEATURE IMPORTANCE (CHECK PENYEBAB) =================
print("\n📊 FITUR PALING BERPENGARUH (Top 5):")
# Kita train RF sederhana pada Text + Stats (Deep terlalu banyak dimensi)
X_exp = np.concatenate([X_stat, X_text], axis=1)
feat_names = ['RMS', 'Centroid', 'Contrast', 'Tempo', 
              'Anger', 'Disgust', 'Fear', 'Joy', 'Neutral', 'Sadness', 'Surprise']

rf_exp = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_exp.fit(X_exp, y)
importances = rf_exp.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(5):
    print(f"   {i+1}. {feat_names[indices[i]]:<10}: {importances[indices[i]]:.4f}")

# ================= 5. MISCLASSIFIED ANALYSIS (HARD EXAMPLES) =================
print("\n💀 DAFTAR LAGU 'BANDEL' (Salah Tebak > 3x di CV):")
# Kita pakai Leave-One-Out kasar dengan CV Predict
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100), X_all, y, cv=5)
misclassified = []

for i in range(len(y)):
    if y[i] != y_pred[i]:
        true_lbl = "Sad" if y[i]==0 else "Relaxed"
        pred_lbl = "Sad" if y_pred[i]==0 else "Relaxed"
        
        # Analisis Kenapa Salah
        reason = []
        # Cek Lirik
        sad_score = X_text[i][5] # Index 5 is sadness
        joy_score = X_text[i][3] # Index 3 is joy
        
        if true_lbl == "Relaxed" and sad_score > 0.5:
            reason.append(f"Lirik Sedih ({sad_score:.2f})")
        if true_lbl == "Sad" and joy_score > 0.5:
            reason.append(f"Lirik Happy ({joy_score:.2f})")
            
        print(f"   ❌ {filenames[i]}")
        print(f"      Label: {true_lbl} -> Pred: {pred_lbl}")
        if reason: print(f"      Penyebab Potensial: {', '.join(reason)}")
        misclassified.append(filenames[i])

print(f"\nTotal Error: {len(misclassified)}/{len(y)}")
print("="*50)