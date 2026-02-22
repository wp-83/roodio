import os
import logging

# Mute Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics_cleaned.csv'
TARGET_MOODS = ['sad', 'relaxed']  # <--- FOKUS KITA SEKARANG
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 10
SEED = 42

print(f"🚀 MEMULAI EXP 28: STAGE 2B STACKING (SAD vs RELAXED)...")

# --- 1. SETUP DATA ---
if not os.path.exists(LYRICS_PATH):
    print("❌ File CSV Lirik tidak ditemukan.")
    exit()

try:
    df = pd.read_csv(LYRICS_PATH, sep=';')
    if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',')
    df.columns = df.columns.str.strip().str.lower()
except: exit()

# Filter Data: Hanya Sad dan Relaxed
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()
print(f"📋 Total Data (Sad + Relaxed): {len(df)}")

# --- 2. LOAD MODELS ---
print("⏳ Loading Models (YAMNet & RoBERTa)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 3. FEATURE EXTRACTION ---
X_audio_features = []
X_text_scores = [] 
y_labels = []
titles_log = []

print("🧠 Extracting Features...")

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        # 1. YAMNet (Semantic Audio)
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # 2. Physics (RMS/ZCR)
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # 3. Tonnetz (PENTING BUAT SAD vs RELAXED)
        # Tonnetz mendeteksi Harmoni (Major vs Minor)
        # Sad biasanya Minor, Relaxed biasanya Major
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harmonic, sr=sr), axis=1)
        
        return np.concatenate([yamnet_vec, [rms, zcr], tonnetz])
    except: return None

def get_text_scores(lyrics):
    try:
        if len(str(lyrics)) < 2: return [0.5, 0.5] # Handle empty
        
        output = nlp_classifier(str(lyrics))[0]
        scores = {item['label']: item['score'] for item in output}
        
        # --- TEAM LOGIC KHUSUS STAGE 2B ---
        # Team Relaxed (Positive/Neutral)
        # Neutral sangat penting disini karena lagu santai seringkali netral
        s_relaxed = scores.get('joy', 0) + scores.get('neutral', 0) + scores.get('surprise', 0)
        
        # Team Sad (Negative)
        s_sad = (scores.get('sadness', 0) + 
                 scores.get('fear', 0) + 
                 scores.get('disgust', 0) + 
                 scores.get('anger', 0))
        
        return [s_sad, s_relaxed] # [Prob Sad, Prob Relaxed]
    except: return [0.5, 0.5]

# LOOP DATA MATCHING
files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])

# Ambil lirik dari DF
lyrics_sad = df[df['mood']=='sad']['lyrics'].tolist()
lyrics_relaxed = df[df['mood']=='relaxed']['lyrics'].tolist()

# Process Sad (Label 0)
for i in tqdm(range(min(len(files_sad), len(lyrics_sad))), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_sad[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(0) 
        titles_log.append(files_sad[i])

# Process Relaxed (Label 1)
for i in tqdm(range(min(len(files_relaxed), len(lyrics_relaxed))), desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_relaxed[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(1)
        titles_log.append(files_relaxed[i])

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) 
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 4. STACKING TRAINING ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
meta_weights = [] 

print(f"\n🚀 START STACKING TRAINING (SAD vs RELAXED)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    # Split
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # 1. Base Model Audio (RF)
    clf_audio = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf_audio.fit(X_aud_tr, y_tr)
    
    # Prediksi Jujur untuk Meta-Train (CV Predict)
    prob_audio_train_cv = cross_val_predict(clf_audio, X_aud_tr, y_tr, cv=3, method='predict_proba')
    prob_audio_test = clf_audio.predict_proba(X_aud_ts)
    
    # 2. Meta Learner Input
    # [Audio_Sad, Audio_Relaxed, Text_Sad, Text_Relaxed]
    X_meta_train = np.concatenate([prob_audio_train_cv, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # 3. Meta Learner Train (Logistic Regression)
    meta_clf = LogisticRegression()
    meta_clf.fit(X_meta_train, y_tr)
    meta_weights.append(meta_clf.coef_[0])
    
    # Predict
    y_pred_fold = meta_clf.predict(X_meta_test)
    
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)

# --- 5. REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR STAGE 2B (STACKING)")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Accuracy: {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Stage 2B Result\nAccuracy: {final_acc:.2f}%')
plt.savefig('cm_exp28_stage2b.png')
plt.show()

# ANALISIS BOBOT
avg_weights = np.mean(meta_weights, axis=0)
print("\n⚖️ ANALISIS KEPUTUSAN META-LEARNER:")
print(f"   Bobot Audio (Tonnetz+YAMNet) : {abs(avg_weights[0]) + abs(avg_weights[1]):.4f}")
print(f"   Bobot Teks (RoBERTa)         : {abs(avg_weights[2]) + abs(avg_weights[3]):.4f}")

if abs(avg_weights[2]) > abs(avg_weights[0]):
    print("👉 Model lebih mengandalkan MAKNA LIRIK.")
else:
    print("👉 Model lebih mengandalkan UNSUR MUSIK (Harmoni/Nada).")