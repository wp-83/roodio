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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
TARGET_COUNT = 40  # Top 35 per kelas (Total 70 Data)
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 42-SIMPLE: RFE + 5-FOLD CV (DETAILED REPORT)...")

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

# --- 2. AUDIO ELITE SELECTION ---
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

print(f"✅ Terpilih {len(selected_files)} Lagu Elite.")

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
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy() # 1024 fitur
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1) # 12 fitur
        rms = np.mean(librosa.feature.rms(y=y)) # 1 fitur
        
        return np.concatenate([yamnet_vec, chroma, [rms]])
    except: return None

for i, path in enumerate(tqdm(selected_files)):
    feat = extract_full_audio(path)
    if feat is not None:
        X_train_final.append(feat)
        y_train_final.append(selected_labels[i])

X_train_final = np.array(X_train_final)
y_train_final = np.array(y_train_final)

# --- 4. RFE (RECURSIVE FEATURE ELIMINATION) ---
print(f"\n🔍 START RFE (Eliminasi Fitur Sampah)...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=SEED)
# Ambil 30 fitur terbaik
rfe = RFE(estimator=rf_selector, n_features_to_select=30, step=0.05) 
X_rfe = rfe.fit_transform(X_train_final, y_train_final)
print(f"   Fitur Akhir: {X_rfe.shape[1]} fitur terpilih!")

# --- 5. TRAINING 5-FOLD (NO LOOP) ---
print(f"\n🚀 START 5-FOLD CROSS VALIDATION...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

# Model Final Config
clf = RandomForestClassifier(
    n_estimators=300,        
    max_depth=None,          
    min_samples_split=5,
    random_state=SEED
)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_rfe, y_train_final)):
    # Split Data
    X_tr, X_ts = X_rfe[train_idx], X_rfe[test_idx]
    y_tr, y_ts = y_train_final[train_idx], y_train_final[test_idx]
    
    # Train
    clf.fit(X_tr, y_tr)
    
    # Predict
    y_pred = clf.predict(X_ts)
    
    # Score
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    
    # Tampilkan Score per Fold
    print(f"   👉 Fold {fold+1}: {acc*100:.2f}%")
    
    # Simpan untuk Matrix
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)

# --- 6. REPORT & VISUALIZATION ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print(f"🏆 HASIL AKHIR (RFE - 5 FOLD)")
print("="*50)

# A. Statistik
print(f"✅ Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")
print("-" * 50)

# B. Classification Report
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

# C. Confusion Matrix Plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sad', 'Relaxed'], 
            yticklabels=['Sad', 'Relaxed'])
plt.title(f'RFE 5-Fold Result\nMean Acc: {mean_acc:.1f}% ±{std_acc:.1f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# --- 7. SAVE FINAL MODEL ---
if not os.path.exists('models'): os.makedirs('models')

# Train ulang di SEMUA data sebelum save
clf.fit(X_rfe, y_train_final)

joblib.dump(rfe, 'models/stage2b_rfe_selector.pkl') 
joblib.dump(clf, 'models/stage2b_rf_rfe.pkl')

print("\n✅ Model Final Disimpan:")
print("   - Selector (Wajib): models/stage2b_rfe_selector.pkl")
print("   - Classifier      : models/stage2b_rf_rfe.pkl")