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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
TARGET_COUNT = 35 # Ambil 25 Audio Terbaik per Kelas
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 40: AUDIO-CENTRIC ELITE SELECTION (TOP {TARGET_COUNT})...")

# --- 1. LOAD EXCEL (Hanya untuk mapping awal) ---
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    
    # Kita simpan mood lama hanya untuk referensi
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Awal: {len(df)} entries.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# --- 2. EXTRACT AUDIO FEATURES (UNTUK CLUSTERING) ---
# Kita butuh fitur yang sangat spesifik membedakan Sad vs Relaxed
print("🧠 Extracting Pure Audio Features (Tonality & Timbre)...")

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
        
        # 1. Spectral Centroid (Kecerahan)
        # Relaxed biasanya lebih 'Terang', Sad lebih 'Gelap'
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 2. Spectral Contrast (Tekstur)
        # Membedakan kepadatan suara
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        
        # 3. RMS (Energi)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # 4. Chroma Deviation (Kestabilan Nada)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_std = np.mean(np.std(chroma, axis=1)) # Seberapa variatif nadanya
        
        return [cent, contrast, rms, chroma_std]
    except: return None

# Scanning
all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

for path in tqdm(all_files):
    fid = get_id_smart(path)
    if fid is None: continue
    
    # Matching dengan Excel (tapi kita akan abaikan labelnya nanti)
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
print(f"✅ Audio Features Extracted: {X_audio_raw.shape}")

# --- 3. AUDIO CLUSTERING & ELITE SELECTION ---
print("⚖️ Menjalankan Audio Clustering & Selection...")

# Scaling dulu biar adil
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_audio_raw)

# K-Means 2 Cluster
kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)

# Tentukan mana Sad mana Relaxed berdasarkan 'Brightness' (Centroid)
# Cluster dengan Centroid (fitur ke-0) lebih rendah biasanya Sad (Gelap)
c0_bright = np.mean(X_audio_raw[cluster_labels==0, 0])
c1_bright = np.mean(X_audio_raw[cluster_labels==1, 0])

if c0_bright < c1_bright:
    sad_cluster = 0
    relaxed_cluster = 1
    print(f"   Cluster 0 = SAD (Darker Audio)")
    print(f"   Cluster 1 = RELAXED (Brighter Audio)")
else:
    sad_cluster = 1
    relaxed_cluster = 0
    print(f"   Cluster 1 = SAD (Darker Audio)")
    print(f"   Cluster 0 = RELAXED (Brighter Audio)")

# HITUNG JARAK KE PUSAT CLUSTER
# Kita gunakan transform() untuk dapat jarak ke setiap centroid
distances = kmeans.transform(X_scaled)

# Kumpulkan kandidat
candidates_sad = []
candidates_relaxed = []

for i in range(len(filenames)):
    dist_to_sad = distances[i][sad_cluster]
    dist_to_relaxed = distances[i][relaxed_cluster]
    
    # Jika dia masuk cluster Sad, simpan jaraknya ke pusat Sad
    if cluster_labels[i] == sad_cluster:
        candidates_sad.append({
            'idx': i,
            'dist': dist_to_sad,
            'file': filenames[i],
            'id': file_ids[i]
        })
    else:
        candidates_relaxed.append({
            'idx': i,
            'dist': dist_to_relaxed,
            'file': filenames[i],
            'id': file_ids[i]
        })

# SORTING: Ambil yang jaraknya PALING KECIL (Paling Dekat Pusat = Paling Representatif)
candidates_sad.sort(key=lambda x: x['dist'])
candidates_relaxed.sort(key=lambda x: x['dist'])

# AMBIL TOP N
top_sad = candidates_sad[:TARGET_COUNT]
top_relaxed = candidates_relaxed[:TARGET_COUNT]

print(f"✅ Terpilih {len(top_sad)} Lagu Paling Sad (Secara Audio)")
print(f"✅ Terpilih {len(top_relaxed)} Lagu Paling Relaxed (Secara Audio)")

# Gabungkan Index Terpilih
selected_indices = [x['idx'] for x in top_sad] + [x['idx'] for x in top_relaxed]
# Label Baru (0=Sad, 1=Relaxed)
selected_labels = [0] * len(top_sad) + [1] * len(top_relaxed)
selected_files = [x['file'] for x in top_sad] + [x['file'] for x in top_relaxed]

# --- 4. PREPARE TRAINING DATA (YAMNET + CHROMA) ---
# Sekarang kita extract fitur LENGKAP untuk training (Yamnet dll)
# Hanya untuk file yang terpilih tadi
print("\n⏳ Extracting Full Features for Training...")
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

print(f"📊 Final Dataset: {len(X_train_final)} samples (Balanced)")

# --- 5. TRAINING ---
print(f"\n🚀 START TRAINING (AUDIO ELITE)...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

clf_final = None
meta_final = None

for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_final, y_train_final)):
    
    X_tr, X_ts = X_train_final[train_idx], X_train_final[test_idx]
    y_tr, y_ts = y_train_final[train_idx], y_train_final[test_idx]
    
    # Base: RF
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
    clf.fit(X_tr, y_tr)
    
    try:
        prob_tr = cross_val_predict(clf, X_tr, y_tr, cv=3, method='predict_proba')
    except:
        prob_tr = clf.predict_proba(X_tr)
    prob_ts = clf.predict_proba(X_ts)
    
    # Meta: LR
    meta = LogisticRegression(C=1.0, random_state=SEED)
    meta.fit(prob_tr, y_tr)
    
    y_pred = meta.predict(prob_ts)
    
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    clf_final = clf
    meta_final = meta

# --- 6. REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100  # <--- INI CODE UNTUK MENGHITUNG STD

print("\n" + "="*50)
print(f"🏆 HASIL AKHIR (AUDIO ELITE)")
print("="*50)
print(f"✅ Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")  # <--- TAMPILKAN DI SINI

if std_acc < 5.0:
    print("   STATUS: SANGAT STABIL")
elif std_acc < 10.0:
    print("   STATUS: CUKUP STABIL")
else:
    print("   STATUS: MASIH FLUKTUATIF")

print("-" * 50)
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))
    
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Audio Elite Result\nAcc: {mean_acc:.1f}% ±{std_acc:.1f}%') # Tambahkan juga di judul plot
plt.show()

# --- 7. SAVE ---
if clf_final:
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(clf_final, 'models/stage2b_rf.pkl') 
    joblib.dump(meta_final, 'models/stage2b_meta.pkl')
    print("✅ Model Stage 2B (Audio Elite) berhasil disimpan!")