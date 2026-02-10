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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

print(f"🚀 MEMULAI EXP 27: STACKING ENSEMBLE (META-LEARNER)...")

# --- 1. SETUP DATA ---
# Load CSV
df = pd.read_csv(LYRICS_PATH, sep=';')
if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',')
df.columns = df.columns.str.strip().str.lower()
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

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
        output = nlp_classifier(str(lyrics))[0]
        scores = {item['label']: item['score'] for item in output}
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0) + scores.get('sadness', 0)
        return [s_angry, s_happy] 
    except: return [0.5, 0.5]

# LOOP DATA MATCHING
files_angry = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'angry')) if f.endswith(('wav','mp3'))])
files_happy = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'happy')) if f.endswith(('wav','mp3'))])
lyrics_angry = df[df['mood']=='angry']['lyrics_clean'].tolist()
lyrics_happy = df[df['mood']=='happy']['lyrics_clean'].tolist()

# Process Angry
for i in tqdm(range(min(len(files_angry), len(lyrics_angry))), desc="Angry"):
    path = os.path.join(RAW_DATA_DIR, 'angry', files_angry[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_angry[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(0) # 0 = Angry
        titles_log.append(files_angry[i])

# Process Happy
for i in tqdm(range(min(len(files_happy), len(lyrics_happy))), desc="Happy"):
    path = os.path.join(RAW_DATA_DIR, 'happy', files_happy[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_happy[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(1) # 1 = Happy
        titles_log.append(files_happy[i])

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) 
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 3. STACKING TRAINING ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
meta_weights = [] # Kita simpan bobot yang dipelajari Meta Learner

print(f"\n🚀 START STACKING TRAINING ({FOLDS}-Fold)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    # A. Split Data Utama
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Text Score Training & Testing (Sudah ada nilainya, tinggal split)
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # --- LEVEL 1: BASE MODELS ---
    
    # 1. Audio Model (Random Forest)
    clf_audio = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf_audio.fit(X_aud_tr, y_tr)
    
    # PENTING: Untuk training Meta-Learner, kita butuh prediksi Audio pada data TRAINING.
    # Kita pakai cross_val_predict biar tidak curang (overfitting)
    # Output: Probabilitas [Angry, Happy] untuk setiap data training
    prob_audio_train_cv = cross_val_predict(clf_audio, X_aud_tr, y_tr, cv=3, method='predict_proba')
    
    # Prediksi Audio untuk data TESTING (Normal)
    prob_audio_test = clf_audio.predict_proba(X_aud_ts)
    
    # --- LEVEL 2: META LEARNER PREPARATION ---
    
    # Gabungkan (Stacking) fitur untuk Meta-Learner
    # Input Meta-Train: [Prob_Audio_Angry, Prob_Audio_Happy, Prob_Text_Angry, Prob_Text_Happy]
    X_meta_train = np.concatenate([prob_audio_train_cv, X_txt_tr], axis=1)
    
    # Input Meta-Test
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # --- LEVEL 3: TRAIN META LEARNER ---
    
    meta_clf = LogisticRegression()
    meta_clf.fit(X_meta_train, y_tr)
    
    # Simpan bobot (Coefficients) untuk analisis nanti
    # Bobot menunjukkan seberapa percaya Meta-Learner pada Audio vs Text
    meta_weights.append(meta_clf.coef_[0]) 
    
    # Final Prediction
    y_pred_fold = meta_clf.predict(X_meta_test)
    
    # Score
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)

# --- 4. REPORT & ANALYSIS ---
print("\n" + "="*50)
print("📊 HASIL AKHIR STACKING")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Accuracy: {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['angry', 'happy']))

# Plot CM
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Angry','Happy'], yticklabels=['Angry','Happy'])
plt.title(f'Stacking Result\nAccuracy: {final_acc:.2f}%')
plt.savefig('cm_exp27_stacking.png')
plt.show()

# --- 5. ANALISIS BOBOT META LEARNER ---
print("\n⚖️ SIAPA YANG LEBIH DIPERCAYA META-LEARNER?")
# Rata-rata bobot dari 10 Fold
avg_weights = np.mean(meta_weights, axis=0)
# Urutan fitur di X_meta: [Audio_Angry, Audio_Happy, Text_Angry, Text_Happy]
# Kita lihat magnitude (nilai mutlak) untuk melihat pengaruhnya

print(f"   Audio Contribution: {abs(avg_weights[0]) + abs(avg_weights[1]):.4f}")
print(f"   Text Contribution : {abs(avg_weights[2]) + abs(avg_weights[3]):.4f}")

if abs(avg_weights[2]) > abs(avg_weights[0]):
    print("👉 Meta-Learner lebih percaya pada LIRIK (Text).")
else:
    print("👉 Meta-Learner lebih percaya pada SUARA (Audio).")