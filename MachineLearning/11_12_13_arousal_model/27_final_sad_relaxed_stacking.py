import os
import re
import logging
import warnings

# --- 1. CONFIGURATION & SETUP ---
# Mute Warnings & Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

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

# PATHS
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics1.csv'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI FINAL STACKING: SAD vs RELAXED (LEAKAGE-FREE)")
print("="*60)

# --- 2. DATA LOADING & CLEANING ---
if not os.path.exists(LYRICS_PATH):
    print(f"❌ Error: File {LYRICS_PATH} tidak ditemukan!")
    exit()

# Load CSV (Auto-detect separator)
try:
    df = pd.read_csv(LYRICS_PATH, sep=';', engine='python')
    if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',', engine='python')
except:
    print("❌ Gagal membaca CSV.")
    exit()

# Filter Target Moods
df.columns = df.columns.str.strip().str.lower()
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

print(f"📊 Total Data: {len(df)} sampel")

# FUNGSI CLEANING (High Quality - Keep Punctuation)
def clean_lyrics_text(text):
    if pd.isna(text) or text == '': return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text) # Hapus metadata [Chorus]
    text = re.sub(r'\(.*?\)', ' ', text)
    garbage = ['lyrics', 'embed', 'contributors', 'translation']
    for w in garbage: text = text.replace(w, '')
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Pertahankan tanda baca (.,!?) untuk intonasi emosi
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['lyrics_clean'] = df['lyrics'].apply(clean_lyrics_text)
print("✅ Cleaning Lirik Selesai.")

# --- 3. MODEL LOADING ---
print("⏳ Loading AI Models (YAMNet & RoBERTa)...")
try:
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)
except Exception as e:
    print(f"❌ Gagal load model: {e}")
    exit()

# --- 4. FEATURE EXTRACTION ---
X_audio_features = []
X_text_scores = [] 
y_labels = []

# Fungsi Ekstraksi Audio
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

# Fungsi Ekstraksi Teks (Logic Sad vs Relaxed)
def get_text_scores(lyrics):
    try:
        lyrics_chunk = str(lyrics)[:512]
        output = nlp_classifier(lyrics_chunk)[0]
        scores = {item['label']: item['score'] for item in output}
        
        # LOGIKA KHUSUS:
        # Sad = Sadness + Fear + Anger + Disgust (Negatif High Arousal/Valence)
        s_sad = scores.get('sadness', 0) + scores.get('fear', 0) + scores.get('anger', 0) + scores.get('disgust', 0)
        
        # Relaxed = Neutral + Joy + Surprise (Positif/Netral Low Arousal)
        s_relaxed = scores.get('neutral', 0) + scores.get('joy', 0) + scores.get('surprise', 0)
        
        return [s_sad, s_relaxed] 
    except: return [0.5, 0.5]

print("🧠 Extracting Features...")

# Loop per Mood
for label_code, mood in enumerate(TARGET_MOODS): # 0: sad, 1: relaxed
    mood_path = os.path.join(RAW_DATA_DIR, mood)
    files = sorted([f for f in os.listdir(mood_path) if f.endswith(('wav','mp3'))])
    lyrics_list = df[df['mood']==mood]['lyrics_clean'].tolist()
    
    limit = min(len(files), len(lyrics_list))
    for i in tqdm(range(limit), desc=mood.capitalize()):
        path = os.path.join(mood_path, files[i])
        
        aud = extract_audio(path)
        txt = get_text_scores(lyrics_list[i])
        
        if aud is not None:
            X_audio_features.append(aud)
            X_text_scores.append(txt)
            y_labels.append(label_code)

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) 
y_labels = np.array(y_labels)

print(f"✅ Siap Training: {X_audio_features.shape[0]} Sampel")

# --- 5. STACKING TRAINING (SAFE METHOD) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
meta_weights = []

print(f"\n🚀 START TRAINING (K-FOLD CV)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    # Split Data Utama
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Split Data Teks (Sudah berupa skor, tinggal split)
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # --- LEVEL 1: BASE MODEL (AUDIO) ---
    clf_audio = RandomForestClassifier(n_estimators=150, random_state=SEED, class_weight='balanced')
    
    # A. Generate Probabilitas Training (ANTI-LEAKAGE)
    # Kita pakai cross_val_predict untuk mendapatkan prediksi 'jujur' pada data training
    prob_audio_train_cv = cross_val_predict(clf_audio, X_aud_tr, y_tr, cv=3, method='predict_proba')
    
    # B. Train Model Audio untuk Testing
    clf_audio.fit(X_aud_tr, y_tr)
    prob_audio_test = clf_audio.predict_proba(X_aud_ts)
    
    # --- LEVEL 2: META LEARNER PREPARATION ---
    # Gabung: [Prob_Audio_Sad, Prob_Audio_Relaxed, Score_Text_Sad, Score_Text_Relaxed]
    X_meta_train = np.concatenate([prob_audio_train_cv, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # --- LEVEL 3: META LEARNER ---
    meta_clf = LogisticRegression(random_state=SEED)
    meta_clf.fit(X_meta_train, y_tr)
    
    # Simpan Bobot (Untuk analisis siapa yang dominan)
    meta_weights.append(meta_clf.coef_[0]) 
    
    # Predict & Evaluasi
    y_pred_fold = meta_clf.predict(X_meta_test)
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)
    
    print(f"   Fold {fold+1}: {acc*100:.1f}%")

# --- 6. FINAL REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*60)
print("📊 HASIL AKHIR: SAD vs RELAXED")
print("="*60)
print(f"🏆 Average Accuracy : {mean_acc:.2f}%")
print(f"📉 Stability (Std)  : ±{std_acc:.2f}%")
print("-" * 60)
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

# Visualisasi Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Final Stacking Result\nAcc: {mean_acc:.1f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('final_sad_relaxed_result.png')
plt.show()

# Analisis Bobot
avg_weights = np.mean(meta_weights, axis=0)
print("\n⚖️ META-LEARNER WEIGHTS (Siapa yang dipercaya?)")
print(f"   🔊 Audio Influence: {abs(avg_weights[0]) + abs(avg_weights[1]):.4f}")
print(f"   📝 Text Influence : {abs(avg_weights[2]) + abs(avg_weights[3]):.4f}")
print("="*60)