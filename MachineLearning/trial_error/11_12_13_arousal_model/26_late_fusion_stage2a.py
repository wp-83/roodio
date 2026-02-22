import os
import logging

# Mute Warning
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics_cleaned.csv'
TARGET_MOODS = ['angry', 'happy'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 26: LATE FUSION (AUDIO + RoBERTa)...")

# --- 1. SETUP DATA ---
# Load CSV
df = pd.read_csv(LYRICS_PATH, sep=';')
if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',')
df.columns = df.columns.str.strip().str.lower()
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

# Load YAMNet
print("⏳ Loading Audio Model (YAMNet)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load RoBERTa
print("⏳ Loading Text Model (RoBERTa)...")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 2. FEATURE EXTRACTION ---
X_audio_features = []
X_text_scores = []  # Kita simpan probabilitas dari RoBERTa di sini
y_labels = []
titles_log = []

print("🧠 Extracting Features (Audio + Text)...")

# Helper Audio
def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        # YAMNet
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        # Physics
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, [rms, zcr]])
    except: return None

# Helper Text (Team Logic)
def get_text_scores(lyrics):
    try:
        output = nlp_classifier(str(lyrics))[0]
        scores = {item['label']: item['score'] for item in output}
        
        # Team Happy
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        # Team Angry
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0) + scores.get('sadness', 0)
        
        return [s_angry, s_happy] # [Prob Angry, Prob Happy]
    except:
        return [0.5, 0.5] # Neutral fallback

# LOOP DATA
# Kita loop berdasarkan CSV agar Audio dan Text sinkron
# Asumsi: Kita cari file audio di folder angry/happy yang namanya mirip atau kita load manual via folder
# Strategi Aman: Load folder audio, cari liriknya di dataframe

files_angry = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'angry')) if f.endswith(('wav','mp3'))])
files_happy = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'happy')) if f.endswith(('wav','mp3'))])

# Kita ambil lirik berdasarkan urutan (seperti Exp 23) karena user confirm datanya urut
lyrics_angry = df[df['mood']=='angry']['lyrics_clean'].tolist()
lyrics_happy = df[df['mood']=='happy']['lyrics_clean'].tolist()

# Angry Processing
for i in tqdm(range(min(len(files_angry), len(lyrics_angry))), desc="Processing Angry"):
    path = os.path.join(RAW_DATA_DIR, 'angry', files_angry[i])
    lyric = lyrics_angry[i]
    
    aud_feat = extract_audio(path)
    txt_score = get_text_scores(lyric)
    
    if aud_feat is not None:
        X_audio_features.append(aud_feat)
        X_text_scores.append(txt_score)
        y_labels.append(0) # 0 = Angry
        titles_log.append(files_angry[i])

# Happy Processing
for i in tqdm(range(min(len(files_happy), len(lyrics_happy))), desc="Processing Happy"):
    path = os.path.join(RAW_DATA_DIR, 'happy', files_happy[i])
    lyric = lyrics_happy[i]
    
    aud_feat = extract_audio(path)
    txt_score = get_text_scores(lyric)
    
    if aud_feat is not None:
        X_audio_features.append(aud_feat)
        X_text_scores.append(txt_score)
        y_labels.append(1) # 1 = Happy
        titles_log.append(files_happy[i])

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) # Shape (N, 2) -> Kolom 0: Angry Prob, Kolom 1: Happy Prob
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 3. FUSION & EVALUATION ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

print(f"\n🚀 START FUSION TRAINING ({FOLDS}-Fold)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    # Split
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Text Scores tidak perlu ditraining ulang, kita cuma butuh ambil bagian test-nya
    # Tapi kita butuh bagian train-nya jika kita mau train Fusion Layer (Meta-Learner)
    # Disini kita pakai Simple Weighted Average dulu biar robust
    
    X_txt_ts = X_text_scores[test_idx]
    
    # 1. Train Audio Classifier (Random Forest)
    # RF bagus untuk data high-dimensi tapi jumlah sampel sedikit
    clf_audio = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf_audio.fit(X_aud_tr, y_tr)
    
    # 2. Predict Audio Probabilities
    # Output: [[Prob_Angry, Prob_Happy], ...]
    prob_audio = clf_audio.predict_proba(X_aud_ts) 
    
    # 3. FUSION LOGIC (WEIGHTED AVERAGE)
    # Kita beri bobot. Misal Audio 60%, Text 40% (Karena audio biasanya lebih jujur soal energi)
    # Atau 50:50. Mari coba 50:50.
    
    W_AUDIO = 0.6
    W_TEXT = 0.4
    
    # Gabung Probabilitas: (Wa * Audio) + (Wt * Text)
    # Note: X_txt_ts sudah dalam format [Prob_Angry, Prob_Happy]
    final_prob = (W_AUDIO * prob_audio) + (W_TEXT * X_txt_ts)
    
    # Tentukan Kelas (0 atau 1)
    y_pred_fold = np.argmax(final_prob, axis=1)
    
    # Score
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)

# --- 4. REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR FUSION (AUDIO + RoBERTa)")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Accuracy: {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['angry', 'happy']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Angry','Happy'], yticklabels=['Angry','Happy'])
plt.title(f'Fusion Result\nAccuracy: {final_acc:.2f}%')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.savefig('cm_exp26_fusion.png')
plt.show()