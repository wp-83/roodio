import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import joblib 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Setup NLTK
try: nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: nltk.download('vader_lexicon')

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI EXP 36: CLEAN & RETRAIN (SAD vs RELAXED)...")

# --- 1. SETUP DATA ---
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    
    lyrics_map = pd.Series(df.lyrics.values, index=df.id).to_dict()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Awal: {len(df)} entries.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

def clean_lyrics(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    return text

sia = SentimentIntensityAnalyzer()

# --- 2. FILTERING DATA (MEMBUANG DATA KONTRADIKTIF) ---
# Aturan Cleaning:
# - Relaxed harus punya VADER > -0.05 (Tidak boleh terlalu negatif)
# - Sad harus punya VADER < 0.05 (Tidak boleh terlalu positif)

valid_ids = []
dropped_count = 0

print("🧹 Melakukan Data Cleaning (VADER Logic)...")
for fid, mood in mood_map.items():
    lyric = clean_lyrics(lyrics_map.get(fid, ""))
    score = sia.polarity_scores(lyric)['compound']
    
    is_valid = True
    if mood == 'relaxed' and score < -0.2: # Relaxed tapi lirik negatif -> BUANG
        is_valid = False
    elif mood == 'sad' and score > 0.2:    # Sad tapi lirik positif -> BUANG
        is_valid = False
        
    if is_valid:
        valid_ids.append(fid)
    else:
        dropped_count += 1
        # print(f"   Drop ID {fid} ({mood}) karena VADER score: {score}")

print(f"✅ Data Bersih: {len(valid_ids)} (Dibuang: {dropped_count})")
print("   Model hanya akan dilatih dengan data yang konsisten!")

# --- 3. EXTRACT FEATURES (Hanya untuk Valid IDs) ---
# Gunakan YAMNet + Chroma (Fitur Terbaik sejauh ini)
print("⏳ Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

X_features = []
y_labels = []

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
        rms = np.mean(librosa.feature.rms(y=y))
        
        return np.concatenate([yamnet_vec, chroma, [rms]])
    except: return None

def get_text_sentiment(lyrics):
    # Kita pakai VADER score sebagai fitur teks utama
    s = sia.polarity_scores(lyrics)
    return [s['compound'], s['pos'], s['neg']]

def get_id(path):
    return os.path.basename(path).split('_')[0].strip()

all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

for path in tqdm(all_files):
    fid = get_id(path)
    if fid not in valid_ids: continue # Skip jika tidak valid
    
    mood = mood_map[fid]
    lyric = lyrics_map[fid]
    label = 0 if mood == 'sad' else 1
    
    aud = extract_audio(path)
    txt = get_text_sentiment(lyric)
    
    if aud is not None:
        # Gabung Audio + Text
        feat = np.concatenate([aud, txt])
        X_features.append(feat)
        y_labels.append(label)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

# --- 4. TRAINING (STACKING) ---
print(f"\n🚀 START TRAINING ON CLEAN DATA ({len(y_labels)} samples)...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

# Model
clf_final = None
meta_final = None

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Base: Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
    clf.fit(X_tr, y_tr)
    
    prob_tr = cross_val_predict(clf, X_tr, y_tr, cv=3, method='predict_proba')
    prob_ts = clf.predict_proba(X_ts)
    
    # Meta: Logistic Regression
    # Karena fiturnya sudah gabungan (Audio+Text di awal), kita bisa langsung predict
    # Tapi biar konsisten stacking, kita pakai probabilitas RF sebagai input Meta
    
    meta = LogisticRegression(C=0.5, random_state=SEED)
    meta.fit(prob_tr, y_tr)
    
    y_pred = meta.predict(prob_ts)
    
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    clf_final = clf
    meta_final = meta

# --- 5. REPORT ---
mean_acc = np.mean(acc_scores) * 100
print("\n" + "="*50)
print(f"🏆 HASIL AKHIR (DATA BERSIH): {mean_acc:.2f}%")
print("="*50)
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Clean Data Result\nAcc: {mean_acc:.1f}%')
plt.show()

# --- 6. SAVE ---
if clf_final:
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(clf_final, 'models/stage2b_rf_clean.pkl')
    joblib.dump(meta_final, 'models/stage2b_meta_clean.pkl')
    print("✅ Model Bersih tersimpan!")