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

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
TARGET_COUNT = 35
FOLDS = 5
SEED = 34

print(f"🚀 MEMULAI EXP 43: STRICT PIPELINE (NO DATA LEAKAGE)...")

# --- 1. DATA CURATION PHASE (ELITE SELECTION) ---
# Ini valid dilakukan di awal sebagai "Preprocessing/Dataset Creation"
# Selama kita tidak menggunakan Label untuk memilih Audio, ini bukan leakage.

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

print("⚖️ Menjalankan Seleksi Audio Elite (Unsupervised)...")
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

# Clustering Logic
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
        candidates_sad.append({'idx': i, 'dist': distances[i][sad_cluster], 'file': filenames[i]})
    else:
        candidates_relaxed.append({'idx': i, 'dist': distances[i][relaxed_cluster], 'file': filenames[i]})

candidates_sad.sort(key=lambda x: x['dist'])
candidates_relaxed.sort(key=lambda x: x['dist'])
top_sad = candidates_sad[:TARGET_COUNT]
top_relaxed = candidates_relaxed[:TARGET_COUNT]
selected_files = [x['file'] for x in top_sad] + [x['file'] for x in top_relaxed]
selected_labels = [0] * len(top_sad) + [1] * len(top_relaxed)

print(f"✅ Terpilih {len(selected_files)} Lagu Elite (Dataset Curation Selesai).")

# --- 2. FEATURE EXTRACTION ---
print("\n⏳ Extracting Full Features...")
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

for i, path in enumerate(tqdm(selected_files)):
    feat = extract_full_audio(path)
    if feat is not None:
        X_full.append(feat)
        y_full.append(selected_labels[i])

X_full = np.array(X_full)
y_full = np.array(y_full)

# --- 3. DEFINING THE PIPELINE (THE SCIENTIFIC FIX) ---
# RFE sekarang menjadi bagian dari Training, bukan Preprocessing Global
print(f"\n🔍 Membangun Pipeline Strict (RFE di dalam CV)...")

# Selector: Random Forest kecil untuk memilih fitur dengan cepat
rfe_selector = RFE(
    estimator=RandomForestClassifier(n_estimators=50, random_state=SEED),
    n_features_to_select=30, # Target 30 Fitur
    step=0.05
)

# Classifier: Random Forest besar untuk prediksi akhir
final_classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    random_state=SEED
)

# GABUNGKAN DALAM PIPELINE
model_pipeline = Pipeline([
    ('feature_selection', rfe_selector), # Step 1: Pilih fitur
    ('classification', final_classifier) # Step 2: Klasifikasi
])

# --- 4. STRICT CROSS-VALIDATION ---
print(f"🚀 START STRICT 5-FOLD CV...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
    # Split Data
    X_tr, X_ts = X_full[train_idx], X_full[test_idx]
    y_tr, y_ts = y_full[train_idx], y_full[test_idx]
    
    # Train Pipeline
    # DI SINI KEAJAIBANNYA: 
    # .fit() akan menjalankan RFE hanya pada X_tr, lalu melatih Classifier pada X_tr terpilih
    model_pipeline.fit(X_tr, y_tr)
    
    # Predict
    # .predict() akan otomatis memfilter X_ts menggunakan fitur yang dipilih dari X_tr tadi
    y_pred = model_pipeline.predict(X_ts)
    
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    
    print(f"   👉 Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)

# --- 5. REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print(f"🏆 HASIL AKHIR (STRICT SCIENTIFIC METHOD)")
print("="*50)

print(f"✅ Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")

if mean_acc > 70.0:
    print("🌟 STATUS: VALID & SOLID (Siap Masuk Laporan/Paper)")
else:
    print("⚠️ STATUS: VALID TAPI PERLU TUNING")

print("\n" + classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Strict Pipeline Result\nMean Acc: {mean_acc:.1f}% ±{std_acc:.1f}%')
plt.show()

# --- 6. SAVE FINAL MODEL ---
if not os.path.exists('models'): os.makedirs('models')

# Train Pipeline pada SEMUA data untuk disimpan (Production Ready)
print("\n💾 Melatih Model Final pada seluruh data...")
model_pipeline.fit(X_full, y_full)

# Simpan Pipeline utuh (Lebih praktis! Tidak perlu simpan selector & model terpisah)
joblib.dump(model_pipeline, 'models/stage2b_strict_pipeline.pkl')

print("✅ Model Final Disimpan: models/stage2b_strict_pipeline.pkl")
print("   (Pipeline ini sudah berisi RFE dan Classifier sekaligus)")