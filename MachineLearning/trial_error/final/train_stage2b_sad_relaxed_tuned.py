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
from sklearn.svm import SVC # <--- GANTI KE SVM
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler # <--- SVM BUTUH SCALING
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI TUNING STAGE 2B: SVM STACKING (SAD vs RELAXED)...")

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

# --- 2. FEATURE EXTRACTION (UPGRADED) ---
X_audio_features = []
X_text_scores = [] 
y_labels = []

print("🧠 Extracting Features (Full Text Emotion)...")

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # Tambah Std Dev biar lebih detail bedain dinamika Sad vs Relaxed
        yamnet_std = tf.math.reduce_std(emb, axis=0).numpy()
        
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, yamnet_std, [rms, zcr]])
    except: return None

def get_text_scores_full(lyrics):
    try:
        # KITA AMBIL SEMUA 7 EMOSI (Bukan dirangkum)
        # Agar SVM bisa milih sendiri mana yang penting
        chunk = str(lyrics)[:512]
        output = nlp_classifier(chunk)[0]
        # Urutkan abjad agar konsisten: anger, disgust, fear, joy, neutral, sadness, surprise
        scores = {item['label']: item['score'] for item in output}
        sorted_keys = sorted(scores.keys()) 
        return [scores[k] for k in sorted_keys]
    except: return [0.0] * 7

def get_id_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

# Loop Files
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
    txt = get_text_scores_full(lyric) # Returns 7 scores
    
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(label)

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_audio_features.shape}")

# --- 3. TRAINING WITH SVM META-LEARNER ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

# Variabel Model Final
clf_audio_final = None
meta_clf_final = None
scaler_final = None # Kita butuh scaler juga

print(f"\n🚀 START SVM STACKING...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # 1. Base Model Audio (Random Forest)
    # RF tidak butuh scaling, jadi aman
    clf_audio = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED)
    clf_audio.fit(X_aud_tr, y_tr)
    
    prob_audio_train = cross_val_predict(clf_audio, X_aud_tr, y_tr, cv=3, method='predict_proba')
    prob_audio_test = clf_audio.predict_proba(X_aud_ts)
    
    # 2. Meta Features Formulation
    # Gabung: [Prob_Sad, Prob_Relaxed, Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise]
    # Total 9 Fitur
    X_meta_train = np.concatenate([prob_audio_train, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # 3. Scaling Meta Features (WAJIB UNTUK SVM)
    # SVM sensitif jika range angka probabilitas (0-1) dan emosi beda
    scaler = StandardScaler()
    X_meta_train_sc = scaler.fit_transform(X_meta_train)
    X_meta_test_sc = scaler.transform(X_meta_test)
    
    # 4. Meta Learner: SVM (RBF Kernel)
    # SVM mencari 'Hyperplane' terbaik untuk memisahkan data
    meta_clf = SVC(kernel='rbf', C=2.0, probability=True, random_state=SEED)
    meta_clf.fit(X_meta_train_sc, y_tr)
    
    y_pred = meta_clf.predict(X_meta_test_sc)
    
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    # Simpan model terakhir
    clf_audio_final = clf_audio
    meta_clf_final = meta_clf
    scaler_final = scaler

# --- 4. REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (SVM TUNED)")
print("="*50)
print(f"🏆 Avg Accuracy : {mean_acc:.2f}%")
print(f"📉 Deviation    : ±{std_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'SVM Stacking Result\nAcc: {mean_acc:.1f}%')
plt.savefig('cm_exp29_tuned.png')
plt.show()

# --- 5. SAVE MODEL ---
print("\n💾 Saving Models...")
if clf_audio_final:
    if not os.path.exists('models'): os.makedirs('models')
    
    joblib.dump(clf_audio_final, 'models/stage2b_rf.pkl')
    joblib.dump(meta_clf_final, 'models/stage2b_meta.pkl')
    # PENTING: Kita juga harus simpan Scaler karena SVM butuh data discale dulu!
    joblib.dump(scaler_final, 'models/stage2b_scaler.pkl') 
    
    print("✅ Model Stage 2B (RF + SVM + Scaler) berhasil disimpan!")