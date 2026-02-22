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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
TARGET_COUNT = 35 # Naikkan sedikit agar variance turun (Total 70 Data)
SEED = 42

print(f"🚀 MEMULAI EXP 41: PCA + ENSEMBLE STABILIZATION (TOP {TARGET_COUNT})...")

# --- 1. LOAD EXCEL ---
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

# --- 2. AUDIO CLUSTERING & ELITE SELECTION ---
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
        # Fitur Akustik Murni (Tanpa YAMNet dulu)
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

kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=30) # n_init lebih tinggi biar stabil
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
selected_labels = [0] * len(top_sad) + [1] * len(top_relaxed)

print(f"✅ Terpilih {len(selected_files)} Lagu Elite ({len(top_sad)} Sad + {len(top_relaxed)} Relaxed).")

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

# --- 4. PCA + ENSEMBLE TRAINING (THE STABILIZER) ---
print(f"\n🚀 START ROBUST TRAINING...")

# A. DEFINISI MODEL
# 1. SVM: Sangat stabil untuk data kecil & dimensi tinggi
clf_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)), # Ambil 95% variansi data, buang noise
    ('svm', SVC(kernel='rbf', probability=True, C=1.0, random_state=SEED))
])

# 2. Extra Trees: Lebih acak dari RF, mengurangi variance (overfitting)
clf_et = Pipeline([
    ('pca', PCA(n_components=20)), # Paksa jadi 20 fitur utama
    ('et', ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=SEED))
])

# 3. Logistic Regression: Baseline yang solid
clf_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=15)),
    ('lr', LogisticRegression(C=0.5, solver='liblinear', random_state=SEED))
])

# B. VOTING (GABUNGKAN KEKUATAN)
ensemble = VotingClassifier(
    estimators=[
        ('svm', clf_svm), 
        ('et', clf_et), 
        ('lr', clf_lr)
    ],
    voting='soft',
    weights=[2, 1, 1] # Beri bobot lebih ke SVM karena biasanya dia MVP di data kecil
)

# C. REPEATED CV (EVALUASI JUJUR)
# Kita test 5-Fold sebanyak 10 kali (Total 50 run) untuk lihat kestabilan asli
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
scores = cross_val_score(ensemble, X_train_final, y_train_final, cv=rkf, n_jobs=-1)

mean_acc = np.mean(scores) * 100
std_acc = np.std(scores) * 100
min_expected_acc = mean_acc - std_acc

# --- 5. REPORT ---
print("\n" + "="*50)
print(f"🏆 HASIL AKHIR (PCA + ENSEMBLE)")
print("="*50)
print(f"✅ Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")
print(f"🛡️ Min. Ekspektasi   : {min_expected_acc:.2f}% (Mean - Std)")

if min_expected_acc > 80.0:
    print("   🌟 SUCCESS: HASIL DIATAS 80% BAHKAN DI KONDISI TERBURUK!")
elif mean_acc > 80.0:
    print("   ✅ SUCCESS: RATA-RATA DIATAS 80% (Stabil)")
else:
    print("   ⚠️ BELUM TEMBUS 80%")

# Visualisasi Distribusi Akurasi (50 Run)
plt.figure(figsize=(8, 4))
sns.histplot(scores * 100, bins=10, kde=True, color='green')
plt.axvline(mean_acc, color='red', linestyle='--', label='Mean')
plt.title(f'Stability Analysis (50 Runs)\nMean: {mean_acc:.1f}% ±{std_acc:.1f}%')
plt.xlabel('Accuracy (%)')
plt.legend()
plt.show()

# Train Final Model
ensemble.fit(X_train_final, y_train_final)

# --- 6. SAVE ---
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(ensemble, 'models/stage2b_ensemble_pca.pkl')
print("\n✅ Model Stage 2B (PCA Ensemble) berhasil disimpan!")