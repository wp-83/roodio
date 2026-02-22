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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 42 # Ganti seed biar fresh

print(f"🚀 MEMULAI EXP 31: DUAL-STREAM EXPERT (SAD vs RELAXED)...")

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

# --- 2. FEATURE EXTRACTION ---
X_audio_features = []
X_text_scores = [] 
y_labels = []

print("🧠 Extracting Features...")

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        # Tambahan fitur statistik
        yamnet_std = tf.math.reduce_std(emb, axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, yamnet_std, [rms, zcr]])
    except: return None

def get_text_scores_full(lyrics):
    try:
        chunk = str(lyrics)[:512]
        output = nlp_classifier(chunk)[0]
        scores = {item['label']: item['score'] for item in output}
        sorted_keys = sorted(scores.keys()) 
        return [scores[k] for k in sorted_keys]
    except: return [0.0] * 7

def get_id_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

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
    txt = get_text_scores_full(lyric)
    
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(label)

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_audio_features.shape}")

# --- 3. DUAL-STREAM STACKING TRAINING ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

# Variabel Model Final
model_audio_final = None
model_text_final = None
model_meta_final = None
scaler_text_final = None # Scaler khusus teks

print(f"\n🚀 START DUAL-STREAM TRAINING...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    # Split Data
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    X_txt_tr, X_txt_ts = X_text_scores[train_idx], X_text_scores[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # --- STREAM 1: AUDIO EXPERT (Random Forest) ---
    clf_audio = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
    clf_audio.fit(X_aud_tr, y_tr)
    
    # Ambil Opini Audio (Probabilitas)
    prob_audio_train = cross_val_predict(clf_audio, X_aud_tr, y_tr, cv=3, method='predict_proba')
    prob_audio_test = clf_audio.predict_proba(X_aud_ts)
    
    # --- STREAM 2: TEXT EXPERT (SVM) ---
    # Kita latih model KHUSUS untuk teks dulu, biar dia pintar sendiri
    scaler_text = StandardScaler()
    X_txt_tr_sc = scaler_text.fit_transform(X_txt_tr)
    X_txt_ts_sc = scaler_text.transform(X_txt_ts)
    
    # Gunakan SVM yang probability=True
    clf_text = SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED)
    clf_text.fit(X_txt_tr_sc, y_tr)
    
    # Ambil Opini Teks (Probabilitas)
    prob_text_train = cross_val_predict(clf_text, X_txt_tr_sc, y_tr, cv=3, method='predict_proba')
    prob_text_test = clf_text.predict_proba(X_txt_ts_sc)
    
    # --- META LEARNER: Gabungkan 2 Opini ---
    # Input Meta: [Prob_Audio_Sad, Prob_Audio_Relaxed, Prob_Text_Sad, Prob_Text_Relaxed]
    X_meta_train = np.concatenate([prob_audio_train, prob_text_train], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, prob_text_test], axis=1)
    
    # Logistic Regression (Voting)
    meta_clf = LogisticRegression(random_state=SEED)
    meta_clf.fit(X_meta_train, y_tr)
    
    # Evaluasi
    y_pred_fold = meta_clf.predict(X_meta_test)
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)
    
    # Simpan Referensi
    model_audio_final = clf_audio
    model_text_final = clf_text
    model_meta_final = meta_clf
    scaler_text_final = scaler_text

# --- 4. REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (DUAL-STREAM)")
print("="*50)
print(f"🏆 Avg Accuracy : {mean_acc:.2f}%")
print(f"📉 Deviation    : ±{std_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Dual-Stream Result\nAcc: {mean_acc:.1f}%')
plt.show()

# Analisis Bobot Voting
weights = model_meta_final.coef_[0] # [Aud0, Aud1, Txt0, Txt1]
aud_weight = abs(weights[0]) + abs(weights[1])
txt_weight = abs(weights[2]) + abs(weights[3])
total = aud_weight + txt_weight
print("\n⚖️ SIAPA YANG LEBIH DIPERCAYA?")
print(f"🔊 Audio Expert Weight : {aud_weight:.4f} ({aud_weight/total*100:.1f}%)")
print(f"📝 Text Expert Weight  : {txt_weight:.4f} ({txt_weight/total*100:.1f}%)")

# --- 5. SAVING MODEL ---
print("\n💾 Saving Models...")
if model_audio_final:
    if not os.path.exists('models'): os.makedirs('models')
    
    joblib.dump(model_audio_final, 'models/stage2b_rf.pkl')     # Audio Expert
    joblib.dump(model_text_final, 'models/stage2b_svm_text.pkl') # Text Expert (NEW)
    joblib.dump(model_meta_final, 'models/stage2b_meta.pkl')    # Boss
    joblib.dump(scaler_text_final, 'models/stage2b_scaler.pkl') # Scaler Text
    
    print("✅ Model Stage 2B (Dual Stream) berhasil disimpan!")