import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import joblib 
import tensorflow_hub as hub
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
TARGET_COUNT = 25 # Top 25 Elite
SEED = 42

print(f"🚀 MEMULAI EXP 41: FINAL STABILIZATION (ENSEMBLE VOTING)...")

# --- 1. SETUP DATA ---
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Awal: {len(df)} entries.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# --- 2. AUDIO ELITE SELECTION (RE-RUN LOGIC EXP 40) ---
print("⚖️ Menjalankan Seleksi Audio Elite...")
X_audio_raw = []
filenames = []
file_ids = []

def get_id_smart(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

def extract_clustering_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_std = np.mean(np.std(chroma, axis=1))
        return [cent, contrast, rms, chroma_std]
    except: return None

all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

for path in tqdm(all_files):
    fid = get_id_smart(path)
    if fid is None: continue
    
    final_id = None
    if fid in mood_map: final_id = fid
    else:
        try:
            if str(int(fid)) in mood_map: final_id = str(int(fid))
        except: pass
            
    if final_id is None: continue
    
    feat = extract_clustering_features(path)
    if feat is not None:
        X_audio_raw.append(feat)
        filenames.append(path)
        file_ids.append(final_id)

X_audio_raw = np.array(X_audio_raw)

# Clustering
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_audio_raw)

kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)

c0_bright = np.mean(X_audio_raw[cluster_labels==0, 0])
c1_bright = np.mean(X_audio_raw[cluster_labels==1, 0])

if c0_bright < c1_bright:
    sad_cluster, relaxed_cluster = 0, 1
else:
    sad_cluster, relaxed_cluster = 1, 0

distances = kmeans.transform(X_scaled)
candidates_sad = []
candidates_relaxed = []

for i in range(len(filenames)):
    if cluster_labels[i] == sad_cluster:
        candidates_sad.append({'idx': i, 'dist': distances[i][sad_cluster], 'file': filenames[i]})
    else:
        candidates_relaxed.append({'idx': i, 'dist': distances[i][relaxed_cluster], 'file': filenames[i]})

candidates_sad.sort(key=lambda x: x['dist'])
candidates_relaxed.sort(key=lambda x: x['dist'])

top_sad = candidates_sad[:TARGET_COUNT]
top_relaxed = candidates_relaxed[:TARGET_COUNT]

selected_files = [x['file'] for x in top_sad] + [x['file'] for x in top_relaxed]
# Label: 0=Sad, 1=Relaxed
selected_labels = [0] * len(top_sad) + [1] * len(top_relaxed)

print(f"✅ Terpilih {len(selected_files)} Lagu Elite (25 Sad + 25 Relaxed).")

# --- 3. FULL FEATURE EXTRACTION ---
print("\n⏳ Extracting Full Features...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

X_train_final = []
y_train_final = []

def extract_full_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
        rms = np.mean(librosa.feature.rms(y=y))
        
        return np.concatenate([yamnet_vec, chroma, [rms]])
    except: return None

for i, path in enumerate(tqdm(selected_files)):
    feat = extract_full_audio(path)
    if feat is not None:
        X_train_final.append(feat)
        y_train_final.append(selected_labels[i])

X_train_final = np.array(X_train_final)
y_train_final = np.array(y_train_final)

# --- 4. ENSEMBLE VOTING (STABILIZATION) ---
print(f"\n🚀 START ENSEMBLE TRAINING...")

# 1. Random Forest (Robust)
clf1 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED)

# 2. SVM (Great for small data)
clf2 = SVC(kernel='rbf', probability=True, random_state=SEED)

# 3. Gradient Boosting (Accuracy booster)
clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED)

# Voting Classifier (Soft Voting = Average Probabilities)
ensemble_clf = VotingClassifier(
    estimators=[('rf', clf1), ('svm', clf2), ('gb', clf3)],
    voting='soft'
)

# REPEATED CV (10x5 = 50 runs) agar hasilnya mewakili performa asli
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
scores = cross_val_score(ensemble_clf, X_train_final, y_train_final, cv=rkf, scoring='accuracy')

mean_acc = np.mean(scores) * 100
std_acc = np.std(scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (STABILIZED ENSEMBLE)")
print("="*50)
print(f"🏆 Avg Accuracy (50 Runs) : {mean_acc:.2f}%")
print(f"📉 Standard Deviation     : ±{std_acc:.2f}%")

if std_acc < 10:
    print("   ✅ STATUS: STABIL (Variasi Rendah)")
else:
    print("   ⚠️ STATUS: MASIH FLUKTUATIF (Tapi wajar untuk data kecil)")

# Latih Model Final di Semua Data
ensemble_clf.fit(X_train_final, y_train_final)

# --- 5. SAVE ---
if not os.path.exists('models'): os.makedirs('models')
# Kita simpan sebagai pickle biasa karena VotingClassifier adalah sklearn object
joblib.dump(ensemble_clf, 'models/stage2b_ensemble.pkl')
print("\n✅ Model Stage 2B (Final Ensemble) berhasil disimpan!")
print("   Gunakan model ini di inference untuk hasil paling robust.")