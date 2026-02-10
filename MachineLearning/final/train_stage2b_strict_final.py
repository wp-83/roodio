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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
TARGET_COUNT = 40 # Ambil 40 Terbaik per Kelas (Total 80 Data)
FOLDS = 5
SEED = 43

print(f"🚀 EXP 42: STRICT RFE PIPELINE (LEAKAGE-FREE VERSION)")
print(f"   Target: {TARGET_COUNT} samples/class | Folds: {FOLDS}")

# ================= 1. DATA LOADING & MAPPING =================
try:
    df = pd.read_excel(LYRICS_PATH)
    # Normalisasi ID: String, Hapus .0, Strip whitespace
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Excel Data Loaded: {len(df)} entries.")
except Exception as e:
    print(f"❌ Error Loading Excel: {e}")
    exit()

# ================= 2. AUDIO ELITE SELECTION (CLUSTERING) =================
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
        # Fitur Akustik Murni
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_std = np.mean(np.std(chroma, axis=1))
        return [cent, contrast, rms, chroma_std]
    except: return None

# Scan Files
all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

# Extract Features
for path in tqdm(all_files, desc="Scanning & Clustering"):
    fid = get_id_smart(path)
    if fid is None: continue
    
    # Robust ID Matching (String/Int)
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

# Clustering Logic
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_audio_raw)

kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=30)
cluster_labels = kmeans.fit_predict(X_scaled)

# Determine Sad/Relaxed Cluster based on Brightness
c0_bright = np.mean(X_audio_raw[cluster_labels==0, 0])
c1_bright = np.mean(X_audio_raw[cluster_labels==1, 0])

if c0_bright < c1_bright: 
    sad_cluster, relaxed_cluster = 0, 1
else: 
    sad_cluster, relaxed_cluster = 1, 0

# Calculate Distance to Center
distances = kmeans.transform(X_scaled)
candidates_sad = []
candidates_relaxed = []

for i in range(len(filenames)):
    if cluster_labels[i] == sad_cluster:
        candidates_sad.append({'dist': distances[i][sad_cluster], 'file': filenames[i]})
    else:
        candidates_relaxed.append({'dist': distances[i][relaxed_cluster], 'file': filenames[i]})

# Sort & Select Top N
candidates_sad.sort(key=lambda x: x['dist'])
candidates_relaxed.sort(key=lambda x: x['dist'])

top_sad = candidates_sad[:TARGET_COUNT]
top_relaxed = candidates_relaxed[:TARGET_COUNT]

selected_files = [x['file'] for x in top_sad] + [x['file'] for x in top_relaxed]
selected_labels = [0] * len(top_sad) + [1] * len(top_relaxed) # 0=Sad, 1=Relaxed

print(f"✅ Elite Selection Complete: {len(selected_files)} files selected.")

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

print(f"📊 Final Dataset Shape: {X_full.shape}")

# ================= 4. STRICT CV (NO DATA LEAKAGE) =================
print(f"\n🚀 START STRICT 5-FOLD CV...")

# Define Pipeline components
rfe_selector = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    n_features_to_select=30, # Ambil 30 fitur emas
    step=0.05
)

classifier = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    random_state=SEED,
    n_jobs=-1
)

# Pipeline: RFE -> Classifier
# Pipeline ensures RFE only sees TRAIN data in each fold
model_pipeline = Pipeline([
    ('feature_selection', rfe_selector),
    ('classification', classifier)
])

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
    X_tr, X_ts = X_full[train_idx], X_full[test_idx]
    y_tr, y_ts = y_full[train_idx], y_full[test_idx]
    
    # Fit Pipeline (RFE learns from X_tr, Classifier learns from RFE(X_tr))
    model_pipeline.fit(X_tr, y_tr)
    
    # Predict (X_ts is transformed by RFE automatically)
    y_pred = model_pipeline.predict(X_ts)
    
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    
    print(f"   👉 Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)

# ================= 5. REPORT =================
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print(f"🏆 HASIL AKHIR (STRICT METHOD)")
print("="*50)
print(f"✅ Mean Accuracy : {mean_acc:.2f}%")
print(f"📉 Std Deviation : ±{std_acc:.2f}%")
print("-" * 50)

print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sad', 'Relaxed'], 
            yticklabels=['Sad', 'Relaxed'])
plt.title(f'Strict Pipeline Result\nAcc: {mean_acc:.1f}% ±{std_acc:.1f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ================= 6. SAVE FINAL MODEL =================
print("\n💾 Training & Saving Final Model (Full Data)...")

# Train pipeline on ALL data
model_pipeline.fit(X_full, y_full)

os.makedirs('models', exist_ok=True)
joblib.dump(model_pipeline, 'models/stage2b_strict_pipeline.pkl')

print("✅ Model Saved: models/stage2b_strict_pipeline.pkl")
print("   (Contains RFE + RF Classifier in one object)")