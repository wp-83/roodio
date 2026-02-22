import os
import logging

# --- 0. ENVIRONMENT SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. CONFIGURATION ---
RAW_DATA_DIR = 'data/raw'
TARGET_MOODS = ['sad', 'relaxed'] 
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 34: STAGE 2B AUDIO ONLY (YAMNET + TONNETZ)...")

# --- 2. MODEL LOADING ---
print("⏳ Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# --- 3. FEATURE EXTRACTION (AUDIO ONLY) ---
X_features = [] 
y_labels = []

def extract_audio_only(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        # 1. YAMNet Embedding (Timbre/Texture) - 1024 dim
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # 2. Tonnetz (Harmoni Major/Minor) - 6 dim
        y_harm = librosa.effects.harmonic(y)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr), axis=1)
        
        # Gabung: 1024 + 6 = 1030 Dimensi
        return np.concatenate([yamnet_vec, tonnetz])
    except: return None

# Load Data
files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])

print("🧠 Extracting Audio Features...")

# SAD (Label 0)
for f in tqdm(files_sad, desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', f)
    feat = extract_audio_only(path)
    if feat is not None:
        X_features.append(feat)
        y_labels.append(0)

# RELAXED (Label 1)
for f in tqdm(files_relaxed, desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', f)
    feat = extract_audio_only(path)
    if feat is not None:
        X_features.append(feat)
        y_labels.append(1)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_features.shape}")

# --- 4. TRAINING (RANDOM FOREST) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
y_true_all = []
y_pred_all = []

print(f"\n🚀 START TRAINING...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Parameter SAMA PERSIS dengan Final Model
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_ts)
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    acc = accuracy_score(y_ts, y_pred)
    print(f"   Fold {fold+1}: {acc*100:.0f}%")

# --- 5. REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR AUDIO ONLY (STAGE 2B)")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Accuracy: {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Audio Only Result\nAccuracy: {final_acc:.2f}%')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.savefig('cm_exp34_audio_only.png')
plt.show()