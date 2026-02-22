import os
import re
import glob
import logging

# Mute Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import joblib 
from transformers import pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 42 # Kita ganti seed sedikit agar fresh

print(f"🚀 MEMULAI EXP 27-FIX: STACKING STAGE 2B (CHROMA ENHANCED)...")

# --- 1. SETUP & CLEANING DATA ---
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    
    lyrics_map = pd.Series(df.lyrics.values, index=df.id).to_dict()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Loaded: {len(df)} entries.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

def clean_lyrics_text(text):
    if pd.isna(text) or text == '': return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for pid in lyrics_map:
    lyrics_map[pid] = clean_lyrics_text(lyrics_map[pid])

print("✅ Cleaning Selesai.")
print("⏳ Loading Models...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 2. FEATURE EXTRACTION (UPDATED WITH CHROMA) ---
X_audio_features = []
X_text_scores = [] 
y_labels = []

print("🧠 Extracting Features (Adding Chroma/Tonality)...")

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        # Pad jika kependekan
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        # 1. YAMNet (Sama seperti Stage 2A)
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # 2. FITUR TAMBAHAN KHUSUS SAD/RELAXED (Chroma & Contrast)
        # Chroma menangkap "Warna Nada" (Major vs Minor)
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
        # Spectral Contrast menangkap "Tekstur Suara"
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        
        # Stats Dasar
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # Gabungkan: [YAMNet(1024) + Chroma(12) + Contrast(7) + RMS(1) + ZCR(1)]
        return np.concatenate([yamnet_vec, chroma, contrast, [rms, zcr]])
    except: return None

def get_text_scores(lyrics):
    try:
        chunk = str(lyrics)[:512]
        output = nlp_classifier(chunk)[0]
        s = {item['label']: item['score'] for item in output}
        
        # Mapping Emosi untuk Sad vs Relaxed
        # Relaxed = Netral/Senang/Terkejut
        s_relaxed = s.get('neutral', 0) + s.get('joy', 0) + s.get('surprise', 0)
        # Sad = Sedih/Takut/Marah/Jijik
        s_sad = s.get('sadness', 0) + s.get('fear', 0) + s.get('anger', 0) + s.get('disgust', 0)
        
        return [s_sad, s_relaxed] 
    except: return [0.5, 0.5]

def get_id_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

# Kumpulkan File
all_audio_files = []
for d in SOURCE_DIRS:
    all_audio_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_audio_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

for file_path in tqdm(all_audio_files):
    fid = get_id_from_filename(file_path)
    if fid not in lyrics_map: continue
    
    mood = mood_map[fid]
    lyric = lyrics_map[fid]
    
    if mood == 'sad': label = 0
    elif mood == 'relaxed': label = 1
    else: continue
    
    aud = extract_audio(file_path)
    txt = get_text_scores(lyric)
    
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(label)

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 3. STABILIZED STACKING TRAINING ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
meta_weights = []

# Model Final
clf_audio_final = None
meta_clf_final = None

print(f"\n🚀 START STACKING TRAINING ({FOLDS}-Fold)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # --- STABILISASI 1: Random Forest (Audio) ---
    # Kita tambah pohon (200) & batasi kedalaman (max_depth=8) agar tidak menghafal data
    clf_audio = RandomForestClassifier(
        n_estimators=200, 
        max_depth=8, 
        min_samples_split=5,
        random_state=SEED
    )
    clf_audio.fit(X_aud_tr, y_tr)
    
    prob_audio_train = cross_val_predict(clf_audio, X_aud_tr, y_tr, cv=3, method='predict_proba')
    prob_audio_test = clf_audio.predict_proba(X_aud_ts)
    
    # Gabung Meta Features
    X_meta_train = np.concatenate([prob_audio_train, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # --- STABILISASI 2: Logistic Regression (Meta) ---
    # C=0.5 membuat model lebih 'konservatif' (tidak gampang berubah karena outlier)
    meta_clf = LogisticRegression(C=0.5, solver='liblinear', random_state=SEED)
    meta_clf.fit(X_meta_train, y_tr)
    
    meta_weights.append(meta_clf.coef_[0]) 
    y_pred_fold = meta_clf.predict(X_meta_test)
    
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)
    
    clf_audio_final = clf_audio
    meta_clf_final = meta_clf

# --- 4. REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR & KESTABILAN (CHROMA ENHANCED)")
print("="*50)
print(f"🏆 Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")
if std_acc < 5.0:
    print("   ✅ STATUS: SANGAT STABIL")
elif std_acc < 10.0:
    print("   ⚠️ STATUS: CUKUP STABIL (Normal)")
else:
    print("   ❌ STATUS: TIDAK STABIL")

print("-" * 50)
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Result (Chroma+Stabilized)\nMean Acc: {mean_acc:.1f}%')
plt.show()

# Analisis Bobot
avg_weights = np.mean(meta_weights, axis=0)
aud_contrib = abs(avg_weights[0]) + abs(avg_weights[1])
txt_contrib = abs(avg_weights[2]) + abs(avg_weights[3])
total = aud_contrib + txt_contrib
print("\n⚖️ ANALISIS KONTRIBUSI")
print(f"👉 Audio (Chroma+YAMNet): {(aud_contrib/total)*100:.1f}%")
print(f"👉 Text (RoBERTa)       : {(txt_contrib/total)*100:.1f}%")

# --- 5. SAVE ---
print("\n💾 Saving Models...")
if clf_audio_final:
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(clf_audio_final, 'models/stage2b_rf.pkl')
    joblib.dump(meta_clf_final, 'models/stage2b_meta.pkl')
    print("✅ Model Stage 2B berhasil disimpan!")