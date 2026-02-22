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
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
TARGET_COUNT = 40 # Total 80 Data
FOLDS = 5
REPEATS = 10 # Total 50 Runs
SEED = 43

print(f"🚀 EXP 43: ROBUST 50-RUNS VALIDATION")
print(f"   Target: {TARGET_COUNT} samples/class | Total: {FOLDS*REPEATS} Runs")

# ================= 1. DATA LOADING & MAPPING =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Excel Data Loaded: {len(df)} entries.")
except Exception as e:
    print(f"❌ Error Loading Excel: {e}")
    exit()

# ================= 2. AUDIO ELITE SELECTION =================
print("\n⚖️ Running Audio Elite Selection...")

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

for path in tqdm(all_files, desc="Scanning & Clustering"):
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
kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=30)
cluster_labels = kmeans.fit_predict(X_scaled)

c0_bright = np.mean(X_audio_raw[cluster_labels==0, 0])
c1_bright = np.mean(X_audio_raw[cluster_labels==1, 0])
if c0_bright < c1_bright: sad_cluster, relaxed_cluster = 0, 1
else: sad_cluster, relaxed_cluster = 1, 0

distances = kmeans.transform(X_scaled)
candidates_sad = []
candidates_relaxed = []

for i in range(len(filenames)):
    if cluster_labels[i] == sad_cluster:
        candidates_sad.append({'dist': distances[i][sad_cluster], 'file': filenames[i]})
    else:
        candidates_relaxed.append({'dist': distances[i][relaxed_cluster], 'file': filenames[i]})

candidates_sad.sort(key=lambda x: x['dist'])
candidates_relaxed.sort(key=lambda x: x['dist'])

top_sad = candidates_sad[:TARGET_COUNT]
top_relaxed = candidates_relaxed[:TARGET_COUNT]

selected_files = [x['file'] for x in top_sad] + [x['file'] for x in top_relaxed]
selected_labels = [0] * len(top_sad) + [1] * len(top_relaxed)

print(f"✅ Elite Selection: {len(selected_files)} files selected.")

# ================= 3. FULL FEATURE EXTRACTION =================
print("\n⏳ Extracting Full Features (YAMNet + Chroma)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

X_full = []
y_full = []

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

for i, path in enumerate(tqdm(selected_files, desc="Processing Audio")):
    feat = extract_full_audio(path)
    if feat is not None:
        X_full.append(feat)
        y_full.append(selected_labels[i])

X_full = np.array(X_full)
y_full = np.array(y_full)

# ================= 4. ROBUST 50-RUNS CV =================
print(f"\n🚀 STARTING REPEATED CV ({FOLDS} Folds x {REPEATS} Repeats)...")

rfe_selector = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    n_features_to_select=30,
    step=0.05
)

classifier = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    random_state=SEED,
    n_jobs=-1
)

model_pipeline = Pipeline([
    ('feature_selection', rfe_selector),
    ('classification', classifier)
])

# RepeatedStratifiedKFold otomatis melakukan looping 50x
rkf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)

# Kita pakai cross_val_score untuk mendapatkan array berisi 50 skor akurasi
scores = cross_val_score(model_pipeline, X_full, y_full, cv=rkf, n_jobs=-1)

# ================= 5. REPORT =================
mean_acc = np.mean(scores) * 100
std_acc = np.std(scores) * 100
min_acc = np.min(scores) * 100
max_acc = np.max(scores) * 100

print("\n" + "="*50)
print(f"🏆 HASIL AKHIR (ROBUST 50 RUNS)")
print("="*50)
print(f"✅ Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")
print(f"🌊 Range Akurasi     : {min_acc:.2f}% - {max_acc:.2f}%")
print("-" * 50)

# Visualisasi Distribusi Akurasi
plt.figure(figsize=(8, 4))
sns.histplot(scores * 100, bins=10, kde=True, color='green')
plt.axvline(mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.1f}%')
plt.title(f'Stability Analysis (50 Runs)\nMean: {mean_acc:.1f}% ±{std_acc:.1f}%')
plt.xlabel('Accuracy (%)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

if mean_acc > 80.0 and std_acc < 8.0:
    print("🌟 KESIMPULAN: MODEL SANGAT STABIL DAN BERPERFORMA TINGGI.")
elif mean_acc > 75.0:
    print("✅ KESIMPULAN: MODEL CUKUP STABIL UNTUK DEPLOYMENT.")
else:
    print("⚠️ KESIMPULAN: MODEL MASIH PERLU EVALUASI ULANG.")

# ================= 6. SAVE FINAL MODEL =================
print("\n💾 Training & Saving Final Model (Full Data)...")
model_pipeline.fit(X_full, y_full)

if not os.path.exists('models'): os.makedirs('models')
joblib.dump(model_pipeline, 'models/stage2b_strict_pipeline.pkl')
print("✅ Model Final Disimpan: models/stage2b_strict_pipeline.pkl")