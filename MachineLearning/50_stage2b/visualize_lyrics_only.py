import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIG =================
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
SEED = 42

print("🚀 VISUALISASI LIRIK ONLY (SAD vs RELAXED)")
print("   Tujuan: Membuktikan apakah Teks lebih pintar daripada Audio.")

# ================= 1. LOAD DATA =================
try:
    df = pd.read_excel(LYRICS_PATH)
    # Bersihkan ID dan Mood
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    
    # Filter hanya Sad dan Relaxed
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    
    # Hapus baris tanpa lirik
    df = df.dropna(subset=['lyrics'])
    df = df[df['lyrics'].str.strip() != ""]
    
    print(f"📦 Total Data Valid: {len(df)} lirik.")
    print(f"   - Sad    : {len(df[df['mood']=='sad'])}")
    print(f"   - Relaxed: {len(df[df['mood']=='relaxed'])}")
    
except Exception as e:
    print(f"❌ Error loading Excel: {e}")
    exit()

# ================= 2. FEATURE EXTRACTION (RoBERTa) =================
print("\n⏳ Loading RoBERTa Emotion Classifier...")
nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

def get_emotion_vector(text):
    # Cleaning ringan
    text = re.sub(r"[^a-z0-9\s]", '', str(text).lower())
    
    # Inference (Max 512 tokens)
    try:
        output = nlp_classifier(text[:512])[0]
    except:
        return None # Jika error / text kosong
    
    # Kita ambil semua 7 skor emosi sebagai fitur vector
    # Urutan: anger, disgust, fear, joy, neutral, sadness, surprise
    scores = {item['label']: item['score'] for item in output}
    
    # Vector 7 Dimensi
    vec = [
        scores.get('anger', 0),
        scores.get('disgust', 0),
        scores.get('fear', 0),
        scores.get('joy', 0),
        scores.get('neutral', 0),
        scores.get('sadness', 0),
        scores.get('surprise', 0)
    ]
    return vec

print("   Extracting Emotion Vectors...")
features = []
labels = []
processed_count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    vec = get_emotion_vector(row['lyrics'])
    if vec is not None:
        features.append(vec)
        labels.append(row['mood']) # 'sad' atau 'relaxed'
        processed_count += 1

X = np.array(features)
feature_names = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutral', 'Sadness', 'Surprise']

# ================= 3. DIMENSIONALITY REDUCTION =================
print("\n🧮 Menghitung Proyeksi Data...")

# PCA (Linear)
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X)

# t-SNE (Non-Linear) - Perplexity disesuaikan dengan jumlah data
perp = min(30, len(X)-1)
tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X)

# ================= 4. PLOTTING =================
plt.figure(figsize=(16, 10))

# --- PLOT 1: PCA (Sebaran Global) ---
plt.subplot(2, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, style=labels, palette=['red', 'blue'], s=100, alpha=0.7)
plt.title('PCA Lirik: Apakah terpisah secara Linear?')
plt.xlabel('PC 1 (Variansi Utama)')
plt.ylabel('PC 2')
plt.grid(True, alpha=0.3)

# --- PLOT 2: t-SNE (Pengelompokan) ---
plt.subplot(2, 2, 2)
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=labels, style=labels, palette=['red', 'blue'], s=100, alpha=0.7)
plt.title('t-SNE Lirik: Apakah membentuk cluster?')
plt.grid(True, alpha=0.3)

# --- DATAFRAME UNTUK BOXPLOT ---
df_scores = pd.DataFrame(X, columns=feature_names)
df_scores['Label'] = labels

# --- PLOT 3: Skor Sadness (Dominan di Sad?) ---
plt.subplot(2, 2, 3)
sns.boxplot(x='Label', y='Sadness', data=df_scores, palette=['#ffcccc', '#ccccff'])
plt.title('Distribusi Emosi "Sadness" di Lirik')

# --- PLOT 4: Skor Joy + Neutral (Dominan di Relaxed?) ---
# Kita gabung Joy dan Neutral karena Relaxed biasanya kombinasi keduanya
df_scores['Relaxed_Score'] = df_scores['Joy'] + df_scores['Neutral']
plt.subplot(2, 2, 4)
sns.boxplot(x='Label', y='Relaxed_Score', data=df_scores, palette=['#ffcccc', '#ccccff'])
plt.title('Distribusi Emosi "Joy + Neutral" di Lirik')

plt.tight_layout()
plt.savefig('visualisasi_lyrics_only.png')
print("\n✅ Gambar disimpan ke 'visualisasi_lyrics_only.png'")
plt.show()

# ================= 5. AUTOMATIC CHECK =================
sad_vecs = X[np.array(labels)=='sad']
rel_vecs = X[np.array(labels)=='relaxed']
dist = np.linalg.norm(np.mean(sad_vecs, axis=0) - np.mean(rel_vecs, axis=0))

print("\n" + "="*50)
print(f"📏 JARAK ANTAR PUSAT LIRIK (Euclidean): {dist:.4f}")
print("="*50)

# Cek Rata-rata Skor Sadness
mean_sad_in_sad = df_scores[df_scores['Label']=='sad']['Sadness'].mean()
mean_sad_in_rel = df_scores[df_scores['Label']=='relaxed']['Sadness'].mean()

print(f"😭 Rata-rata Skor 'Sadness':")
print(f"   - Di Lagu Sad    : {mean_sad_in_sad:.2f}")
print(f"   - Di Lagu Relaxed: {mean_sad_in_rel:.2f}")
diff = mean_sad_in_sad - mean_sad_in_rel

if diff > 0.3:
    print(f"\n✅ KESIMPULAN: LIRIK ADALAH KUNCI! (Beda Skor: {diff:.2f})")
    print("   Boxplot Sadness harusnya jomplang. Gunakan model Teks-Only atau Stacking.")
elif diff > 0.1:
    print(f"\n⚠️ KESIMPULAN: Lirik membantu, tapi tidak mutlak. (Beda Skor: {diff:.2f})")
else:
    print(f"\n❌ KESIMPULAN: Lirik pun mirip. Dataset ini sangat susah.")