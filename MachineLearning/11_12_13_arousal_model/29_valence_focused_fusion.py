import os
import logging

# Mute Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- KONFIGURASI ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics1.csv'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 29: VALENCE FOCUSED FUSION (HARMONY + SENTIMENT)...")

# --- 1. SETUP DATA ---
if not os.path.exists(LYRICS_PATH):
    print("❌ File CSV tidak ditemukan.")
    exit()

try:
    df = pd.read_csv(LYRICS_PATH, sep=';')
    if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',')
    df.columns = df.columns.str.strip().str.lower()
except: exit()

df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

# --- 2. LOAD NLP MODEL ---
print("⏳ Loading RoBERTa (Sentiment Expert)...")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 3. FEATURE EXTRACTION (VALENCE ONLY) ---
X_features = [] # Gabungan Audio & Teks langsung
y_labels = []
titles_log = []

print("🧠 Extracting VALENCE Features...")

def extract_harmony_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        # Kita potong tengah biar dapet inti lagunya
        duration = len(y)
        center = duration // 2
        # Ambil 30 detik di tengah (atau full kalau kurang)
        segment = y[max(0, center - 16000*15) : min(duration, center + 16000*15)]
        
        # 1. CHROMA (12 Nada Dasar) -> Deteksi Kunci Lagu
        chroma = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr), axis=1)
        
        # 2. SPECTRAL CONTRAST -> Deteksi "Kekasaran" suara (Valence indicator)
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr), axis=1)
        
        # 3. TONNETZ -> Deteksi Hubungan Harmoni (Major/Minor)
        # Ini fitur paling mahal untuk Valence Audio
        y_harm = librosa.effects.harmonic(segment)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr), axis=1)
        
        # Gabung: 12 + 7 + 6 = 25 Fitur Harmoni
        return np.concatenate([chroma, contrast, tonnetz])
    except: return None

def get_sentiment_score(lyrics):
    try:
        if len(str(lyrics)) < 2: return 0.0 # Neutral
        
        output = nlp_classifier(str(lyrics))[0]
        scores = {item['label']: item['score'] for item in output}
        
        # RUMUS VALENCE SEDERHANA
        # Positive = Joy + Relaxed(Neutral/Surprise)
        pos_score = scores.get('joy', 0) + scores.get('neutral', 0) + scores.get('surprise', 0)
        
        # Negative = Sad + Anger + Fear
        neg_score = scores.get('sadness', 0) + scores.get('anger', 0) + scores.get('fear', 0) + scores.get('disgust', 0)
        
        # Kita kembalikan satu angka: Net Sentiment (Range -1 sampai 1)
        # Jika > 0 berarti Relaxed, Jika < 0 berarti Sad
        return [pos_score - neg_score] 
    except: return [0.0]

# LOOP PROCESSING
files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])
lyrics_sad = df[df['mood']=='sad']['lyrics'].tolist()
lyrics_relaxed = df[df['mood']=='relaxed']['lyrics'].tolist()

# SAD (Label 0)
for i in tqdm(range(min(len(files_sad), len(lyrics_sad))), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    aud = extract_harmony_features(path) # Audio Harmony
    txt = get_sentiment_score(lyrics_sad[i]) # Text Sentiment
    
    if aud is not None:
        # Concatenate Audio (25 dim) + Text (1 dim) = 26 Dimensi Valence
        feat = np.concatenate([aud, txt]) 
        X_features.append(feat)
        y_labels.append(0)
        titles_log.append(files_sad[i])

# RELAXED (Label 1)
for i in tqdm(range(min(len(files_relaxed), len(lyrics_relaxed))), desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    aud = extract_harmony_features(path)
    txt = get_sentiment_score(lyrics_relaxed[i])
    
    if aud is not None:
        feat = np.concatenate([aud, txt])
        X_features.append(feat)
        y_labels.append(1)
        titles_log.append(files_relaxed[i])

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_features.shape} (N_Samples, N_Features)")
# Fitur terakhir adalah Skor Sentiment Teks

# --- 4. TRAINING & EVALUATION ---
# Kita pakai Standard Scaler karena Chroma (0-1) beda skala dengan Contrast (bisa puluhan)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
weights_log = []

print(f"\n🚀 START TRAINING (LOGISTIC REGRESSION)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_labels)):
    X_tr, X_ts = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Linear Classifier (Sangat bagus untuk melihat bobot fitur)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_ts)
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    
    # Simpan bobot fitur
    weights_log.append(model.coef_[0])
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    print(f"   Fold {fold+1}: {acc*100:.2f}%")

# --- 5. REPORT ---
print("\n" + "="*50)
print("📊 HASIL STAGE 2B (HARMONY + SENTIMENT)")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Accuracy: {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Valence Fusion\nAccuracy: {final_acc:.2f}%')
plt.savefig('cm_exp29_valence.png')
plt.show()

# --- 6. ANALISIS BOBOT FITUR (KUNCI UTAMA) ---
# Kita ingin membuktikan apakah Audio dan Teks berkontribusi seimbang?
avg_weights = np.mean(weights_log, axis=0)
# Indeks terakhir (-1) adalah Text Sentiment
text_weight = abs(avg_weights[-1])
# Sisanya adalah Audio Harmony
audio_weight_avg = np.mean(np.abs(avg_weights[:-1])) 
audio_weight_total = np.sum(np.abs(avg_weights[:-1]))

print("\n⚖️ KONTRIBUSI FITUR (Feature Importance):")
print(f"   🔹 Rata-rata Bobot Audio (Harmony) : {audio_weight_avg:.4f}")
print(f"   🔹 Bobot Teks (Sentiment Score)    : {text_weight:.4f}")

if text_weight > (audio_weight_avg * 10): # Misal teks 10x lebih kuat
    print("⚠️ HASIL JOMPLANG: Teks terlalu mendominasi.")
elif audio_weight_avg > (text_weight * 10):
    print("⚠️ HASIL JOMPLANG: Audio terlalu mendominasi.")
else:
    print("✅ BALANCE: Kedua modalitas berkontribusi (Tidak Jomplang).")