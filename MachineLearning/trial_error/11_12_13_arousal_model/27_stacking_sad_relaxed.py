import os
import re
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI KHUSUS SAD/RELAXED ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics1.csv'
TARGET_MOODS = ['sad', 'relaxed']  # <--- UBAH TARGET
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43 

print(f"🚀 MEMULAI EXP 27: STACKING ENSEMBLE (SAD vs RELAXED)...")

# --- 1. SETUP & CLEANING DATA ---
# Load CSV
try:
    df = pd.read_csv(LYRICS_PATH, sep=';', engine='python')
    if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',', engine='python')
except FileNotFoundError:
    print(f"❌ File {LYRICS_PATH} tidak ditemukan. Pastikan path benar.")
    exit()

# Normalisasi Header
df.columns = df.columns.str.strip().str.lower()
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

print(f"📊 Data Awal: {len(df)} baris. Melakukan Cleaning...")

# === FUNGSI PREPROCESSING LIRIK ===
def clean_lyrics_text(text):
    if pd.isna(text) or text == '':
        return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text) # Hapus [Chorus]
    text = re.sub(r'\(.*?\)', ' ', text) # Hapus (x2)
    garbage = ['lyrics', 'embed', 'contributors', 'translation']
    for w in garbage:
        text = text.replace(w, '')
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text) # Keep punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['lyrics_clean'] = df['lyrics'].apply(clean_lyrics_text)
print("✅ Lirik berhasil dibersihkan.")

# Load Models
print("⏳ Loading Models (YAMNet & RoBERTa)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 2. FEATURE EXTRACTION ---
X_audio_features = []
X_text_scores = [] 
y_labels = []
titles_log = []

print("🧠 Extracting Features...")

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, [rms, zcr]])
    except: return None

def get_text_scores(lyrics):
    try:
        lyrics_chunk = str(lyrics)[:512]
        output = nlp_classifier(lyrics_chunk)[0]
        scores = {item['label']: item['score'] for item in output}
        
        # --- LOGIKA SKOR BARU (SAD vs RELAXED) ---
        # Relaxed biasanya 'Neutral' atau 'Joy' (kalem)
        s_relaxed = scores.get('neutral', 0) + scores.get('joy', 0) + scores.get('surprise', 0)
        
        # Sad adalah 'Sadness' ditambah emosi negatif lain
        s_sad = scores.get('sadness', 0) + scores.get('fear', 0) + scores.get('anger', 0) + scores.get('disgust', 0)
        
        return [s_sad, s_relaxed] 
    except: return [0.5, 0.5]

files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])
lyrics_sad = df[df['mood']=='sad']['lyrics_clean'].tolist()
lyrics_relaxed = df[df['mood']=='relaxed']['lyrics_clean'].tolist()

# Process Sad (Label 0)
for i in tqdm(range(min(len(files_sad), len(lyrics_sad))), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_sad[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(0) # 0 = SAD
        titles_log.append(files_sad[i])

# Process Relaxed (Label 1)
for i in tqdm(range(min(len(files_relaxed), len(lyrics_relaxed))), desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_relaxed[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(1) # 1 = RELAXED
        titles_log.append(files_relaxed[i])

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) 
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 3. STACKING TRAINING ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
meta_weights = []

print(f"\n🚀 START STACKING TRAINING ({FOLDS}-Fold)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # Base Model: Audio
    clf_audio = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf_audio.fit(X_aud_tr, y_tr)
    
    prob_audio_train_cv = cross_val_predict(clf_audio, X_aud_tr, y_tr, cv=3, method='predict_proba')
    prob_audio_test = clf_audio.predict_proba(X_aud_ts)
    
    # Meta Learner Input
    X_meta_train = np.concatenate([prob_audio_train_cv, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # Meta Learner: Logistic Regression
    meta_clf = LogisticRegression()
    meta_clf.fit(X_meta_train, y_tr)
    
    meta_weights.append(meta_clf.coef_[0]) 
    y_pred_fold = meta_clf.predict(X_meta_test)
    
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)

# --- 4. REPORT & ANALYSIS ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100 

print("\n" + "="*50)
print("📊 HASIL AKHIR & KESTABILAN (SAD vs RELAXED)")
print("="*50)

print(f"🏆 Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")
if std_acc < 5.0:
    print("   ✅ STATUS: SANGAT STABIL")
elif std_acc < 10.0:
    print("   ⚠️ STATUS: CUKUP STABIL (Normal)")
else:
    print("   ❌ STATUS: TIDAK STABIL (Hasil fluktuatif)")

print("-" * 50)
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

# Plot CM
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Stacking Sad vs Relaxed\nMean Acc: {mean_acc:.1f}% ±{std_acc:.1f}%')
plt.savefig('cm_exp27_sad_relaxed.png')
plt.show()

# --- 5. ANALISIS BOBOT ---
print("\n⚖️ ANALISIS KONTRIBUSI FITUR")
avg_weights = np.mean(meta_weights, axis=0)

# Ingat urutannya: [Audio_Sad, Audio_Relaxed, Text_Sad, Text_Relaxed]
audio_contrib = abs(avg_weights[0]) + abs(avg_weights[1])
text_contrib  = abs(avg_weights[2]) + abs(avg_weights[3])

print(f"🔊 Audio Score : {audio_contrib:.4f}")
print(f"📝 Text Score  : {text_contrib:.4f}")

total = audio_contrib + text_contrib
if total > 0:
    print(f"👉 Persentase  : Audio {(audio_contrib/total)*100:.1f}% vs Text {(text_contrib/total)*100:.1f}%")
else:
    print("👉 Model bingung (Total bobot mendekati 0)")

import joblib
joblib.dump(clf_audio, 'models/stage2b_rf.pkl')
joblib.dump(meta_clf, 'models/stage2b_meta.pkl')
print("💾 Model Stage 2B berhasil disimpan!")