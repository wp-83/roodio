import os
import glob
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow_hub as hub
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
TARGET_COUNT = 40  # Top 40 per class
SEED = 43

print("🚀 RETRAINING STAGE 2B (STRICT SUPERVISED SELECTION)")
print("   Fixing the 'Flipped Label' issue...")

# ================= 1. FUNGSI SELECTOR PER FOLDER =================
# Kita tidak lagi mencampur file. Kita cari elite per folder.

def extract_cluster_feat(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        return [
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
            np.mean(librosa.feature.rms(y=y)),
            np.mean(np.std(librosa.feature.chroma_cens(y=y, sr=sr), axis=1))
        ]
    except: return None

def get_elite_files_from_mood(mood_name, n_select=40):
    print(f"\n🔍 Scanning Elite Files for: {mood_name.upper()}...")
    files = []
    for d in SOURCE_DIRS:
        # Cari spesifik di folder mood tersebut
        files.extend(glob.glob(os.path.join(d, mood_name, '*.wav')))
        files.extend(glob.glob(os.path.join(d, mood_name, '*.mp3')))
    
    files.sort() # Konsistensi
    
    # Extract fitur ringkas
    feats = []
    valid_files = []
    for f in tqdm(files, desc=f"Feat Extract {mood_name}"):
        v = extract_cluster_feat(f)
        if v is not None:
            feats.append(v)
            valid_files.append(f)
            
    # Cari Centroid (Pusat Data) folder ini
    X = np.array(feats)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    # Kita pakai KMeans k=1 untuk mencari titik tengah folder ini
    kmeans = KMeans(n_clusters=1, random_state=SEED, n_init=10)
    kmeans.fit(X_sc)
    
    # Hitung jarak setiap file ke titik tengah
    distances = kmeans.transform(X_sc).flatten()
    
    # Ambil N file terdekat dengan pusat (paling representatif)
    sorted_indices = np.argsort(distances)
    elite_files = [valid_files[i] for i in sorted_indices[:n_select]]
    
    print(f"✅ Selected {len(elite_files)} elite samples for {mood_name}")
    return elite_files

# ================= 2. EXECUTE SELECTION =================
sad_files = get_elite_files_from_mood('sad', TARGET_COUNT)
relaxed_files = get_elite_files_from_mood('relaxed', TARGET_COUNT)

# Gabungkan: SAD DULUAN (Label 0), lalu RELAXED (Label 1)
final_files = sad_files + relaxed_files
y_labels = np.array([0] * len(sad_files) + [1] * len(relaxed_files))

print(f"\n📦 Total Dataset: {len(final_files)} files")
print(f"   Label 0 (Sad)    : {len(sad_files)}")
print(f"   Label 1 (Relaxed): {len(relaxed_files)}")

# ================= 3. FULL TRAINING EXTRACTION =================
print("\n⏳ Extracting Full Features (YAMNet + Chroma)...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_full(path):
    y, sr = librosa.load(path, sr=16000)
    y = np.pad(y, (0, max(0, 16000 - len(y))))
    _, emb, _ = yamnet(y)
    return np.concatenate([
        tf.reduce_mean(emb, axis=0).numpy(),
        np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1),
        [np.mean(librosa.feature.rms(y=y))]
    ])

X_train = np.array([extract_full(p) for p in tqdm(final_files)])

# ================= 4. TRAIN & SAVE (WINNING PARAMS) =================
print("\n🔨 Training Model (SVM Linear, C=0.5, 30 Feats)...")

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('selection', RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=SEED), n_features_to_select=30)), 
    ('model', SVC(probability=True, kernel='linear', C=0.5, random_state=SEED)) 
])

# Verifikasi Akurasi Internal
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(pipeline, X_train, y_labels, cv=skf, scoring='accuracy')

print("\n" + "="*50)
print(f"🏆 NEW MODEL ACCURACY (Internal CV)")
print("="*50)
print(f"✅ Mean Accuracy : {np.mean(scores)*100:.2f}%")
print(f"📉 Std Dev       : ±{np.std(scores)*100:.2f}%")
print("-" * 50)

# Save Model
print("💾 Overwriting 'models/stage2b_tuned_final.pkl'...")
pipeline.fit(X_train, y_labels)

if not os.path.exists('models'): os.makedirs('models')
joblib.dump(pipeline, 'models/stage2b_tuned_final.pkl')

print("✅ DONE! Model diperbaiki. Sekarang Label 0 PASTI Sad, Label 1 PASTI Relaxed.")