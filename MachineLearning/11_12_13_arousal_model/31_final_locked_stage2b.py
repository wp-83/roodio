import os
import logging

# --- 0. ENVIRONMENT SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. LOCKED CONFIGURATION ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics.csv'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# 🔒 PARAMETER DIKUNCI
FOLDS = 5 
SEED = 43 
N_ESTIMATORS = 200
CLASS_WEIGHT = 'balanced'

print(f"🚀 MEMULAI FINAL LOCKED ARCHITECTURE (STAGE 2B)...")
print(f"🎯 Target: Valence Classification (Sad vs Relaxed)")
print(f"🚫 Excluded Features: RMS, ZCR, Tempo (Arousal Indicators)")

# --- 2. DATA LOADING ---
if not os.path.exists(LYRICS_PATH):
    print("❌ File CSV tidak ditemukan.")
    exit()

try:
    df = pd.read_csv(LYRICS_PATH, sep=';')
    if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',')
    df.columns = df.columns.str.strip().str.lower()
except: exit()

df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

# --- 3. MODEL LOADING ---
print("⏳ Loading Pretrained Models...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1") # Audio Embedding
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True) # Text Context

# --- 4. FINAL FEATURE EXTRACTION ---
X_features = [] 
y_labels = []

def extract_final_features(audio_path, lyrics_text):
    try:
        # --- A. AUDIO MODALITY (DOMINAN) ---
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        # 1. YAMNet Embedding (Timbre/Texture) - 1024 dim
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # 2. Tonnetz (Harmoni Major/Minor) - 6 dim
        # Anchor utama untuk Valence di musik
        y_harm = librosa.effects.harmonic(y)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr), axis=1)
        
        # ❌ RMS & ZCR DIBUANG (Sesuai Protokol)
        
        # --- B. TEXT MODALITY (AUXILIARY) ---
        # 3. RoBERTa Emotion Scores - 6 dim
        # [joy, neutral, surprise, sadness, fear, anger]
        if pd.isna(lyrics_text) or len(str(lyrics_text)) < 2:
            text_feats = [0.0] * 6 # Handle missing lyrics
        else:
            output = nlp_classifier(str(lyrics_text))[0]
            scores = {item['label']: item['score'] for item in output}
            text_feats = [
                scores.get('joy', 0),
                scores.get('neutral', 0),
                scores.get('surprise', 0),
                scores.get('sadness', 0),
                scores.get('fear', 0),
                scores.get('anger', 0)
            ]
        
        # Gabung: 1024 (Audio) + 6 (Audio) + 6 (Text) = 1036 Dimensi
        return np.concatenate([yamnet_vec, tonnetz, text_feats])
    except Exception as e:
        return None

# Loop Data Loading
files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])
lyrics_sad = df[df['mood']=='sad']['lyrics'].tolist()
lyrics_relaxed = df[df['mood']=='relaxed']['lyrics'].tolist()

print("🧠 Extracting Pure Valence Features...")
# SAD
for i in tqdm(range(min(len(files_sad), len(lyrics_sad))), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    feat = extract_final_features(path, lyrics_sad[i])
    if feat is not None:
        X_features.append(feat)
        y_labels.append(0)

# RELAXED
for i in tqdm(range(min(len(files_relaxed), len(lyrics_relaxed))), desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    feat = extract_final_features(path, lyrics_relaxed[i])
    if feat is not None:
        X_features.append(feat)
        y_labels.append(1)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"✅ Final Feature Vector Shape: {X_features.shape}")
print("   (Harusnya 1036 kolom: 1030 Audio + 6 Text)")

# --- 5. TRAINING (LOCKED CLASSIFIER) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
y_true_all = []
y_pred_all = []
feature_importances_log = []

print(f"\n🚀 START TRAINING (RandomForest seed={SEED})...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # ✅ CLASSIFIER FINAL (DIKUNCI)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        random_state=SEED, 
        class_weight=CLASS_WEIGHT
    )
    clf.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_ts)
    
    # Logging
    feature_importances_log.append(clf.feature_importances_)
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    acc = accuracy_score(y_ts, y_pred)
    print(f"   Fold {fold+1}: {acc*100:.0f}%")

# --- 6. FINAL REPORT ---
print("\n" + "="*50)
print("🏁 STATUS AKHIR: STAGE 2B (FINAL LOCKED)")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"📊 Accuracy : {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

# Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'FINAL STAGE 2B\nAccuracy: {final_acc:.2f}%')
plt.ylabel('True Mood'); plt.xlabel('Predicted Mood')
plt.savefig('cm_final_stage2b_locked.png')
plt.show()

# --- 7. MODALITY CONTRIBUTION CHECK ---
avg_imp = np.mean(feature_importances_log, axis=0)
# Audio: Index 0 sampai 1029 (1024 YAMNet + 6 Tonnetz)
imp_audio = np.sum(avg_imp[:1030])
# Text: Index 1030 sampai 1035
imp_text = np.sum(avg_imp[1030:])

print("\n⚖️ MODALITY CONTRIBUTION CHECK:")
print(f"   🔊 Audio Contribution : {imp_audio*100:.2f}%")
print(f"   📝 Text Contribution  : {imp_text*100:.2f}%")

if imp_audio > imp_text:
    print("✅ VALID: Audio is Dominant (Sesuai Hirarki)")
else:
    print("⚠️ WARNING: Text is Dominant (Cek ulang data audio)")

print("\n" + "="*60)
print("🧾 GOLDEN PARAGRAPH (Untuk Laporan):")
print('Stage 2B focuses on valence classification within the low-arousal music subset produced by Stage 1. A hybrid multimodal architecture is employed, where audio features derived from YAMNet embeddings and Tonnetz-based harmonic descriptors serve as the primary indicators of emotional valence, while lyric-based emotion scores extracted using a pretrained RoBERTa model provide auxiliary semantic context. Classification is performed using a Random Forest model due to its robustness to high-dimensional heterogeneous features and limited data size.')
print("="*60)