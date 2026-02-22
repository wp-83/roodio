import os
import logging

# --- CONFIG ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# PATH BARU (YANG SUDAH BENAR)
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics_cleaned.csv'  
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 37: FINAL FUSION (DATA CORRECTED)...")

# --- 1. LOAD DATA ---
if not os.path.exists(LYRICS_PATH):
    print(f"❌ File {LYRICS_PATH} tidak ditemukan!")
    exit()

# Baca CSV dengan engine python agar fleksibel
df = pd.read_csv(LYRICS_PATH, sep=None, engine='python')
df.columns = df.columns.str.strip().str.lower()
df['mood'] = df['mood'].str.lower().str.strip()

# Filter hanya Sad & Relaxed
df = df[df['mood'].isin(TARGET_MOODS)].copy()

# PENTING: Urutkan data berdasarkan mood (opsional, tapi membantu debug)
# Kita asumsikan urutan baris di CSV sesuai dengan urutan file audio yang di-sort abjad
df_sad = df[df['mood'] == 'sad'].reset_index(drop=True)
df_relaxed = df[df['mood'] == 'relaxed'].reset_index(drop=True)

print(f"✅ Loaded Lyrics: {len(df_sad)} Sad, {len(df_relaxed)} Relaxed")

# --- 2. LOAD MODELS ---
print("⏳ Loading AI Models (YAMNet + RoBERTa)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 3. ADVANCED FEATURE EXTRACTION ---
X_features = [] 
y_labels = []

def get_text_valence(text):
    # Hitung skor sentimen gabungan
    if pd.isna(text) or len(str(text)) < 5: return 0.0
    
    # Karena lirik panjang, kita potong 512 karakter pertama agar cepat
    # (Biasanya mood lagu sudah terlihat di verse 1 & chorus awal)
    text_chunk = str(text)[:512] 
    
    output = nlp_classifier(text_chunk)[0]
    s = {item['label']: item['score'] for item in output}
    
    # Rumus Valence: (Positif) - (Negatif)
    # Relaxed biasanya lebih 'Neutral' atau 'Joy' dibanding Sad yang 'Sadness'
    pos_score = s.get('joy', 0) + s.get('neutral', 0) + s.get('surprise', 0)
    neg_score = s.get('sadness', 0) + s.get('fear', 0) + s.get('anger', 0)
    
    return pos_score - neg_score

def extract_features(audio_path, lyrics_text):
    try:
        # A. AUDIO (Mean + Std)
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        _, emb, _ = yamnet_model(y)
        yam_mean = tf.reduce_mean(emb, axis=0).numpy()
        yam_std = tf.math.reduce_std(emb, axis=0).numpy() # Menangkap dinamika
        
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        ton_mean = np.mean(tonnetz, axis=1)
        ton_std = np.std(tonnetz, axis=1)
        
        # B. TEXT (Valence Score)
        txt_val = get_text_valence(lyrics_text)
        
        # Gabung: [Audio_Mean, Audio_Std, Tonnetz, Text_Valence]
        return np.concatenate([yam_mean, yam_std, ton_mean, ton_std, [txt_val]])
    except Exception as e:
        print(f"⚠️ Error {audio_path}: {e}")
        return None

# --- 4. PROCESSING LOOP (ALIGNMENT) ---
files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])

print("🧠 Extracting Features (Audio + Full Lyrics)...")

# Process SAD
# Asumsi: File audio ke-1 (abjad) cocok dengan Baris ke-1 di CSV (df_sad)
limit_sad = min(len(files_sad), len(df_sad))
for i in tqdm(range(limit_sad), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    lyric = df_sad.iloc[i]['lyrics']
    
    feat = extract_features(path, lyric)
    if feat is not None:
        X_features.append(feat)
        y_labels.append(0) # 0 = Sad

# Process RELAXED
limit_relaxed = min(len(files_relaxed), len(df_relaxed))
for i in tqdm(range(limit_relaxed), desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    lyric = df_relaxed.iloc[i]['lyrics']
    
    feat = extract_features(path, lyric)
    if feat is not None:
        X_features.append(feat)
        y_labels.append(1) # 1 = Relaxed

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_features.shape}")

# --- 5. TRAINING (High-End Config) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
y_true_all = []
y_pred_all = []
feature_importances = []

print(f"\n🚀 START TRAINING (Random Forest)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Tuning Random Forest untuk data kecil tapi fitur banyak
    clf = RandomForestClassifier(
        n_estimators=300,        # Banyak pohon biar stabil
        max_depth=10,            # Jangan terlalu dalam (cegah overfitting)
        min_samples_leaf=2,      # Daun minimal 2 (cegah hafal data)
        random_state=SEED, 
        class_weight='balanced'
    )
    clf.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_ts)
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    feature_importances.append(clf.feature_importances_)
    
    acc = accuracy_score(y_ts, y_pred)
    print(f"   Fold {fold+1}: {acc*100:.0f}%")

# --- 6. HASIL AKHIR ---
print("\n" + "="*50)
print("📊 HASIL AKHIR HYBRID FUSION (CORRECTED DATA)")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Accuracy: {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

# Cek Kontribusi Teks
avg_imp = np.mean(feature_importances, axis=0)
text_imp = avg_imp[-1] * 100
print(f"⚖️ Kontribusi Fitur Teks (Valence): {text_imp:.2f}%")

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Final Fusion Result\nAccuracy: {final_acc:.2f}%')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig('cm_exp37_final.png')
plt.show()