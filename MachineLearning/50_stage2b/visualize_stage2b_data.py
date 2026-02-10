import os
import glob
import numpy as np
import pandas as pd  # <--- SUDAH DITAMBAHKAN
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
SEED = 43

print("🚀 VISUALISASI FULL DATA STAGE 2B (SAD vs RELAXED)")
print("   Tujuan: Melihat sebaran SELURUH data (100%) tanpa filter.")

# ================= 1. LOAD ALL FILES =================
sad_files = []
rel_files = []

for d in SOURCE_DIRS:
    sad_files.extend(glob.glob(os.path.join(d, 'sad', '*.wav')) + glob.glob(os.path.join(d, 'sad', '*.mp3')))
    rel_files.extend(glob.glob(os.path.join(d, 'relaxed', '*.wav')) + glob.glob(os.path.join(d, 'relaxed', '*.mp3')))

# Sorting agar rapi
sad_files.sort()
rel_files.sort()

# KITA PAKAI SEMUA DATA (TIDAK ADA PEMOTONGAN [:40])
print(f"📦 Total Data Ditemukan: {len(sad_files)} Sad, {len(rel_files)} Relaxed")

# ================= 2. EKSTRAKSI FITUR =================
print("⏳ Extracting Features (YAMNet + Audio Stats)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def get_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        # Padding standar (Penting!)
        if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
        
        # 1. YAMNet Vector (1024 dimensions)
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # 2. Audio Stats (Manual Features)
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr)) 
        
        return yamnet_vec, [rms, spec_cent, spec_bw, chroma]
    except: return None, None

X_yamnet = []
X_manual = [] # [RMS, Centroid, Bandwidth, Chroma]
labels = []

# Process SAD
for f in tqdm(sad_files, desc="Processing Sad"):
    y_vec, m_vec = get_features(f)
    if y_vec is not None:
        X_yamnet.append(y_vec)
        X_manual.append(m_vec)
        labels.append('Sad')

# Process RELAXED
for f in tqdm(rel_files, desc="Processing Relaxed"):
    y_vec, m_vec = get_features(f)
    if y_vec is not None:
        X_yamnet.append(y_vec)
        X_manual.append(m_vec)
        labels.append('Relaxed')

X_yamnet = np.array(X_yamnet)
X_manual = np.array(X_manual)

# ================= 3. DIMENSIONALITY REDUCTION =================
print("\n🧮 Menghitung PCA & t-SNE...")

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_yamnet)

# PCA
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)

# ================= 4. PLOTTING =================
plt.figure(figsize=(15, 12))

# --- PLOT 1: PCA ---
plt.subplot(2, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, style=labels, palette=['red', 'blue'], s=100, alpha=0.7)
plt.title(f'PCA Analysis (Linear View)\nSeparable?', fontsize=12)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.grid(True, alpha=0.3)

# --- PLOT 2: t-SNE ---
plt.subplot(2, 2, 2)
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=labels, style=labels, palette=['red', 'blue'], s=100, alpha=0.7)
plt.title(f't-SNE Analysis (Non-Linear View)\nGrouping?', fontsize=12)
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.grid(True, alpha=0.3)

# --- DATAFRAME MANUAL ---
# Sekarang variabel 'pd' sudah aman
df_manual = pd.DataFrame(X_manual, columns=['RMS', 'Centroid', 'Bandwidth', 'Chroma'])
df_manual['Label'] = labels

# --- PLOT 3: RMS (Loudness) ---
plt.subplot(2, 2, 3)
sns.boxplot(x='Label', y='RMS', data=df_manual, palette=['#ffcccc', '#ccccff'])
plt.title('Distribusi Volume (RMS)', fontsize=12)

# --- PLOT 4: Spectral Centroid (Brightness) ---
plt.subplot(2, 2, 4)
sns.boxplot(x='Label', y='Centroid', data=df_manual, palette=['#ffcccc', '#ccccff'])
plt.title('Distribusi Kecerahan Suara (Centroid)', fontsize=12)

plt.tight_layout()
plt.savefig('visualisasi_stage2b_full.png')
print("\n✅ Gambar disimpan ke 'visualisasi_stage2b_full.png'")
plt.show()

# ================= 5. AUTOMATIC DIAGNOSIS =================
print("\n" + "="*50)
print("🤖 DIAGNOSA OTOMATIS (DATASET PENUH)")
print("="*50)

sad_pca = X_pca[np.array(labels)=='Sad']
rel_pca = X_pca[np.array(labels)=='Relaxed']
dist_centroid = np.linalg.norm(np.mean(sad_pca, axis=0) - np.mean(rel_pca, axis=0))

print(f"📏 Jarak Antar Pusat (PCA): {dist_centroid:.2f}")

if dist_centroid < 2.0:
    print("⚠️ KESIMPULAN: DATA SANGAT TERCAMPUR (Hard to Separate)")
    print("   -> SVM Linear tidak akan bisa membedakan ini.")
    print("   -> Solusi: Gunakan Random Forest atau XGBoost.")
elif dist_centroid < 5.0:
    print("⚠️ KESIMPULAN: DATA TUMPANG TINDIH (Moderate)")
else:
    print("✅ KESIMPULAN: DATA TERPISAH DENGAN BAIK")