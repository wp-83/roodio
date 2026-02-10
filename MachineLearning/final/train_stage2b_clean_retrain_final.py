import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
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
import tensorflow_hub as hub
import tensorflow as tf

# Setup NLTK
try: nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: nltk.download('vader_lexicon')

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI EXP 36-FINAL: CLEAN & RETRAIN (FIX ID FORMAT)...")

# --- 1. SETUP DATA (EXCEL) ---
try:
    df = pd.read_excel(LYRICS_PATH)
    
    # NORMALISASI ID EXCEL:
    # Kita ubah jadi string, hilangkan .0 (jika ada), dan hilangkan spasi
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    # Filter Mood
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    
    # Mapping
    lyrics_map = pd.Series(df.lyrics.values, index=df.id).to_dict()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    
    print(f"📊 Data Excel Loaded: {len(df)} entries.")
    print(f"   Contoh ID Excel: {list(lyrics_map.keys())[:5]}")
except Exception as e:
    print(f"❌ Error loading Excel: {e}")
    exit()

def clean_lyrics(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    return text

sia = SentimentIntensityAnalyzer()

# --- 2. DATA CLEANING (VADER FILTER) ---
valid_ids = set()
dropped_count = 0

print("🧹 Melakukan Data Cleaning (VADER Logic)...")
for fid, mood in mood_map.items():
    lyric = clean_lyrics(lyrics_map.get(fid, ""))
    score = sia.polarity_scores(lyric)['compound']
    
    is_valid = True
    if mood == 'relaxed' and score < -0.1: # Relaxed tapi lirik negatif
        is_valid = False
    elif mood == 'sad' and score > 0.1:    # Sad tapi lirik positif
        is_valid = False
        
    if is_valid:
        valid_ids.add(fid)
    else:
        dropped_count += 1

print(f"✅ Data Valid (Whitelist): {len(valid_ids)} ID. (Dibuang: {dropped_count})")

# --- 3. MATCHING FILE AUDIO ---
print("⏳ Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

X_features = []
y_labels = []

# Fungsi Ekstraksi ID yang LEBIH PINTAR
def get_id_smart(filename):
    base = os.path.basename(filename)
    # Format: "id_judul.wav" -> Ambil bagian sebelum "_" pertama
    parts = base.split('_')
    if len(parts) > 1:
        raw_id = parts[0].strip()
        # Coba bersihkan jika ada karakter aneh, tapi biarkan angka/huruf
        # Jika ID Excel murni angka (misal '1'), dan file '01', kita handle nanti di pencocokan
        return raw_id
    return None

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
    s = sia.polarity_scores(lyrics)
    return [s['compound'], s['pos'], s['neg']]

# Scanning File
all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))
    # Support typo user (.waf) just in case, walau glob mungkin ga nemu kalau ga diset
    all_files.extend(glob.glob(os.path.join(d, "**", "*.waf"), recursive=True))

print(f"📂 Total File Audio Ditemukan: {len(all_files)}")

# --- DEBUG MATCHING ---
matched_count = 0
examples_file_id = []

print("⚙️ Memulai Pencocokan ID & Ekstraksi...")
for path in tqdm(all_files):
    file_id = get_id_smart(path)
    
    if file_id is None: continue # Skip file tanpa underscore
    
    # LOGIKA PENCOCOKAN YANG FLEKSIBEL
    # 1. Cek Exact Match
    match_found = False
    final_id = None
    
    if file_id in valid_ids:
        match_found = True
        final_id = file_id
    else:
        # 2. Cek Integer Match (misal File "01" == Excel "1")
        try:
            int_id = str(int(file_id)) # "01" -> 1 -> "1"
            if int_id in valid_ids:
                match_found = True
                final_id = int_id
        except:
            pass # Bukan angka
            
    if not match_found:
        if len(examples_file_id) < 3: examples_file_id.append(f"File: {os.path.basename(path)} -> ID Terbaca: '{file_id}' (Gagal Match)")
        continue

    matched_count += 1
    
    # Ambil Data
    mood = mood_map[final_id]
    lyric = lyrics_map[final_id]
    label = 0 if mood == 'sad' else 1
    
    aud = extract_audio(path)
    txt = get_text_sentiment(lyric)
    
    if aud is not None:
        feat = np.concatenate([aud, txt])
        X_features.append(feat)
        y_labels.append(label)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

if len(y_labels) == 0:
    print("\n❌ GAGAL: Masih 0 Sampel!")
    print(f"   Contoh ID Excel Valid: {list(valid_ids)[:3]}")
    print(f"   Contoh ID File Gagal :")
    for ex in examples_file_id: print(f"      {ex}")
    print("👉 PASTIKAN NAMA FILE MENGGUNAKAN UNDERSCORE (_), misal: '1_JudulLagu.wav'")
    exit()

# --- 4. TRAINING ---
print(f"\n🚀 START TRAINING ({len(y_labels)} samples matched)...")

# Adaptive CV
n_splits = min(5, len(y_labels) // 2)
if n_splits < 2: skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
else: skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

acc_scores = []
y_true_all = []
y_pred_all = []

clf_final = None
meta_final = None

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Base: RF
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
    clf.fit(X_tr, y_tr)
    
    # Meta Input
    try:
        prob_tr = cross_val_predict(clf, X_tr, y_tr, cv=min(3, len(y_tr)//2), method='predict_proba')
    except:
        prob_tr = clf.predict_proba(X_tr)
        
    prob_ts = clf.predict_proba(X_ts)
    
    # Meta: LR
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
print(f"🏆 HASIL AKHIR (FIX ID): {mean_acc:.2f}%")
print("="*50)

if len(y_true_all) > 0:
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
    print("✅ Model Stage 2B (Clean & Retrain) berhasil disimpan!")