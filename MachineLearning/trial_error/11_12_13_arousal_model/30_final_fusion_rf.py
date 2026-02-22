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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- KONFIGURASI ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics_cleaned.csv'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5 # Sesuai permintaan
SEED = 43 # Sesuai permintaan

print(f"🚀 MEMULAI EXP 30: HYBRID FUSION (YAMNET + TONNETZ + ROBERTA)...")

# --- 1. SETUP DATA ---
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

# --- 2. LOAD MODELS ---
print("⏳ Loading Models...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 3. FEATURE EXTRACTION ---
X_features = [] 
y_labels = []
feature_names = [] # Untuk tracking nama fitur

print("🧠 Extracting Features...")

def extract_features(audio_path, lyrics_text):
    try:
        # --- A. AUDIO EXTRACTION ---
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        # 1. YAMNet (1024 fitur - Deep Audio)
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy() # 1024 dim
        
        # 2. Tonnetz (6 fitur - Harmoni Musik)
        # Penting untuk membedakan Major (Relaxed) vs Minor (Sad)
        y_harm = librosa.effects.harmonic(y)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr), axis=1) # 6 dim
        
        # --- B. TEXT EXTRACTION ---
        output = nlp_classifier(str(lyrics_text))[0]
        scores = {item['label']: item['score'] for item in output}
        
        # Kita ambil emosi yang relevan sebagai fitur
        # Urutan: [Joy, Neutral, Surprise, Sadness, Fear, Anger]
        text_feats = [
            scores.get('joy', 0),
            scores.get('neutral', 0),
            scores.get('surprise', 0),
            scores.get('sadness', 0),
            scores.get('fear', 0),
            scores.get('anger', 0)
        ]
        
        # Gabung Semua: [YAMNet (1024) + Tonnetz (6) + Text (6)]
        return np.concatenate([yamnet_vec, tonnetz, text_feats])
    except: return None

# LOOP PROCESSING
files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])
lyrics_sad = df[df['mood']=='sad']['lyrics'].tolist()
lyrics_relaxed = df[df['mood']=='relaxed']['lyrics'].tolist()

# SAD (Label 0)
for i in tqdm(range(min(len(files_sad), len(lyrics_sad))), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    feat = extract_features(path, lyrics_sad[i])
    if feat is not None:
        X_features.append(feat)
        y_labels.append(0)

# RELAXED (Label 1)
for i in tqdm(range(min(len(files_relaxed), len(lyrics_relaxed))), desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    feat = extract_features(path, lyrics_relaxed[i])
    if feat is not None:
        X_features.append(feat)
        y_labels.append(1)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_features.shape}")
# Struktur Fitur:
# 0-1023    : YAMNet (Audio Texture)
# 1024-1029 : Tonnetz (Audio Harmony)
# 1030-1035 : RoBERTa (Text Meaning)

# --- 4. TRAINING (RANDOM FOREST) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
feature_importances_log = []

print(f"\n🚀 START TRAINING...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Random Forest sangat tangguh untuk data campuran (High Dim + Low Dim)
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_ts)
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    
    # Simpan Feature Importance
    feature_importances_log.append(clf.feature_importances_)
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    print(f"   Fold {fold+1}: {acc*100:.2f}%")

# --- 5. REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR STAGE 2B (HYBRID RF)")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Accuracy: {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Hybrid Result\nAccuracy: {final_acc:.2f}%')
plt.savefig('cm_exp30_final.png')
plt.show()

# --- 6. ANALISIS JOMPLANG (SCIENTIFIC CHECK) ---
avg_imp = np.mean(feature_importances_log, axis=0)

# Hitung Total Kontribusi per Modalitas
# YAMNet + Tonnetz (0 sampai 1029)
imp_audio = np.sum(avg_imp[:1030])
# Text (1030 sampai selesai)
imp_text = np.sum(avg_imp[1030:])

print("\n⚖️ KONTRIBUSI MODALITAS (Apakah Jomplang?):")
print(f"   🔊 Audio Contribution : {imp_audio*100:.2f}%")
print(f"   📝 Text Contribution  : {imp_text*100:.2f}%")

if abs(imp_audio - imp_text) < 0.3: # Jika beda kurang dari 30%
    print("✅ CUKUP SEIMBANG (Multimodal Works!)")
else:
    print("⚠️ DOMINASI TERDETEKSI (Salah satu modalitas bekerja jauh lebih keras)")