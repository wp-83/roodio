import os
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIG ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics_cleaned.csv' # Gunakan yang sudah dibersihkan
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI EXP 39: BALANCED FUSION (PCA + FULL EMOTIONS)...")

# --- 1. LOAD DATA ---
df = pd.read_csv(LYRICS_PATH, sep=None, engine='python')
df.columns = df.columns.str.strip().str.lower()
df['mood'] = df['mood'].str.lower().str.strip()
df_sad = df[df['mood'] == 'sad'].reset_index(drop=True)
df_relaxed = df[df['mood'] == 'relaxed'].reset_index(drop=True)

# --- 2. LOAD MODELS ---
print("⏳ Loading Models...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 3. FEATURE EXTRACTION (TERPISAH) ---
# Kita simpan Audio dan Teks terpisah dulu untuk diproses PCA
audio_features_raw = []
text_features_raw = []
y_labels = []

def get_text_emotions(text):
    # Mengambil 7 dimensi emosi, bukan cuma 1 valence
    if pd.isna(text) or len(str(text)) < 5: 
        return [0.0] * 7 # Urutan: anger, disgust, fear, joy, neutral, sadness, surprise
    
    text_chunk = str(text)[:512] 
    output = nlp_classifier(text_chunk)[0]
    # Urutkan berdasarkan abjad key agar konsisten
    sorted_scores = [item['score'] for item in sorted(output, key=lambda x: x['label'])]
    return sorted_scores

def get_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        _, emb, _ = yamnet_model(y)
        yam_mean = tf.reduce_mean(emb, axis=0).numpy()
        yam_std = tf.math.reduce_std(emb, axis=0).numpy()
        
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        ton_mean = np.mean(tonnetz, axis=1)
        ton_std = np.std(tonnetz, axis=1)
        
        return np.concatenate([yam_mean, yam_std, ton_mean, ton_std])
    except Exception as e:
        print(f"⚠️ Error {audio_path}: {e}")
        return None

# --- 4. PROCESSING LOOP ---
files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])

print("🧠 Extracting Raw Features...")

# Process SAD
limit_sad = min(len(files_sad), len(df_sad))
for i in tqdm(range(limit_sad), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    lyric = df_sad.iloc[i]['lyrics']
    
    aud = get_audio_features(path)
    txt = get_text_emotions(lyric)
    
    if aud is not None:
        audio_features_raw.append(aud)
        text_features_raw.append(txt)
        y_labels.append(0)

# Process RELAXED
limit_relaxed = min(len(files_relaxed), len(df_relaxed))
for i in tqdm(range(limit_relaxed), desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    lyric = df_relaxed.iloc[i]['lyrics']
    
    aud = get_audio_features(path)
    txt = get_text_emotions(lyric)
    
    if aud is not None:
        audio_features_raw.append(aud)
        text_features_raw.append(txt)
        y_labels.append(1)

# Convert to Numpy
X_audio = np.array(audio_features_raw)
X_text = np.array(text_features_raw)
y = np.array(y_labels)

print(f"📊 Dimensi Awal -> Audio: {X_audio.shape}, Teks: {X_text.shape}")

# --- 5. PREPROCESSING & FUSION (PCA) ---
print("\n🔧 Melakukan Scaling & PCA pada Audio...")

# A. Scale Audio (Wajib sebelum PCA)
scaler_audio = StandardScaler()
X_audio_scaled = scaler_audio.fit_transform(X_audio)

# B. PCA Audio (Compress 2000+ fitur jadi yg penting saja)
# n_components=0.95 artinya: "Simpan fitur yang mewakili 95% variansi data"
pca = PCA(n_components=0.95, random_state=SEED)
X_audio_pca = pca.fit_transform(X_audio_scaled)

print(f"📉 Audio Compressed: {X_audio.shape[1]} fitur -> {X_audio_pca.shape[1]} fitur utama")

# C. Scale Text (Agar range angkanya sebanding dengan PCA Audio)
scaler_text = StandardScaler()
X_text_scaled = scaler_text.fit_transform(X_text)

# D. FUSION (Gabungkan Audio PCA + Teks Lengkap)
X_final = np.concatenate([X_audio_pca, X_text_scaled], axis=1)
print(f"✅ FINAL FUSION SHAPE: {X_final.shape}")

# --- 6. TRAINING ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
feature_importances = []

print(f"\n🚀 START TRAINING (Random Forest)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_final, y)):
    X_tr, X_ts = X_final[train_idx], X_final[test_idx]
    y_tr, y_ts = y[train_idx], y[test_idx]
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED)
    clf.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_ts)
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    feature_importances.append(clf.feature_importances_)
    
    print(f"   Fold {fold+1}: {acc*100:.1f}%")

# --- 7. ANALISIS HASIL ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (PCA BALANCED)")
print("="*50)
print(f"🏆 Mean Accuracy: {mean_acc:.2f}%")
print(f"📉 Stability (Std Dev): ±{std_acc:.2f}% (Semakin kecil semakin stabil)")
print("-" * 30)

# Cek Kontribusi Teks (7 Fitur Terakhir adalah Teks)
avg_imp = np.mean(feature_importances, axis=0)
n_audio_feat = X_audio_pca.shape[1]
text_imp_total = np.sum(avg_imp[n_audio_feat:]) * 100
print(f"⚖️ Kontribusi Total Teks (7 Emosi): {text_imp_total:.2f}%")

# Plot Confusion Matrix (Fold Terakhir sebagai sampel)
cm = confusion_matrix(y_ts, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Sample Confusion Matrix\nText Contrib: {text_imp_total:.1f}%')
plt.savefig('exp39_result.png')
plt.show()