import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
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
TARGET_COUNT = 30 
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 39: ELITE SELECTION (TOP {TARGET_COUNT} PER CLASS)...")

# --- 1. SETUP DATA ---
try:
    df = pd.read_excel(LYRICS_PATH)
    # Normalisasi ID
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
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

# --- 2. ELITE SELECTION (SORTING BY PURITY) ---
candidates = []

print("⚖️ Menilai Kemurnian Data (VADER Score)...")
for fid, mood in mood_map.items():
    lyric = clean_lyrics(lyrics_map.get(fid, ""))
    # Compound score: -1 (Sangat Negatif) s.d +1 (Sangat Positif)
    score = sia.polarity_scores(lyric)['compound']
    
    candidates.append({
        'id': fid,
        'mood': mood,
        'score': score
    })

# Pisahkan
sad_list = [x for x in candidates if x['mood'] == 'sad']
relaxed_list = [x for x in candidates if x['mood'] == 'relaxed']

# SORTING LOGIC:
# Sad Terbaik = Score Paling Rendah (Paling Negatif) -> Ascending
sad_list.sort(key=lambda x: x['score']) 

# Relaxed Terbaik = Score Paling Tinggi (Paling Positif) -> Descending
relaxed_list.sort(key=lambda x: x['score'], reverse=True)

# AMBIL TOP N
top_sad = sad_list[:TARGET_COUNT]
top_relaxed = relaxed_list[:TARGET_COUNT]

print(f"\n🏆 SELEKSI {TARGET_COUNT} TERBAIK:")
print(f"   Sad Paling Murni     : Score {top_sad[0]['score']} s.d {top_sad[-1]['score']}")
print(f"   Relaxed Paling Murni : Score {top_relaxed[0]['score']} s.d {top_relaxed[-1]['score']}")

# Gabungkan ID Valid
valid_ids = set([x['id'] for x in top_sad] + [x['id'] for x in top_relaxed])
print(f"✅ Total Whitelist ID: {len(valid_ids)}")

# --- 3. EXTRACT FEATURES ---
print("⏳ Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

X_features = []
y_labels = []

def get_id_smart(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        # YAMNet (Content)
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # Chroma (Tonality - Penting Sad/Relaxed)
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
        
        # Energy
        rms = np.mean(librosa.feature.rms(y=y))
        
        return np.concatenate([yamnet_vec, chroma, [rms]])
    except: return None

def get_text_sentiment(lyrics):
    s = sia.polarity_scores(lyrics)
    # Kita pakai Compound score sebagai fitur utama
    return [s['compound'], s['pos'], s['neg']]

# Scanning
all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

print("⚙️ Memulai Ekstraksi...")
matched_count = 0

for path in tqdm(all_files):
    file_id = get_id_smart(path)
    if file_id is None: continue 
    
    # Matching Logic
    final_id = None
    if file_id in valid_ids: final_id = file_id
    else:
        try:
            if str(int(file_id)) in valid_ids: final_id = str(int(file_id))
        except: pass
            
    if final_id is None: continue

    matched_count += 1
    mood = mood_map[final_id]
    label = 0 if mood == 'sad' else 1
    
    aud = extract_audio(path)
    txt = get_text_sentiment(lyrics_map[final_id])
    
    if aud is not None:
        feat = np.concatenate([aud, txt])
        X_features.append(feat)
        y_labels.append(label)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

# Cek Keseimbangan Akhir
n_sad = np.sum(y_labels == 0)
n_relaxed = np.sum(y_labels == 1)
print(f"\n📊 Distribusi Final: Sad={n_sad}, Relaxed={n_relaxed}")

if n_sad == 0 or n_relaxed == 0:
    print("❌ Gawat! Salah satu kelas kosong setelah matching file audio.")
    print("   Cek apakah file audio untuk Top 25 ID benar-benar ada di folder.")
    exit()

# --- 4. TRAINING (STACKING) ---
print(f"\n🚀 START TRAINING (ELITE BALANCED)...")

# Kita pakai 5-Fold. Karena datanya pas 50 (ideal), tiap fold ada 10 sampel (5 Sad, 5 Relaxed)
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

clf_final = None
meta_final = None

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Base: Random Forest
    # Kita tidak perlu class_weight='balanced' karena data SUDAH balanced (25 vs 25)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
    clf.fit(X_tr, y_tr)
    
    # Meta Input
    try:
        prob_tr = cross_val_predict(clf, X_tr, y_tr, cv=3, method='predict_proba')
    except:
        prob_tr = clf.predict_proba(X_tr)
        
    prob_ts = clf.predict_proba(X_ts)
    
    # Meta: Logistic Regression
    meta = LogisticRegression(C=1.0, random_state=SEED) # C=1.0 standar
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
print(f"🏆 HASIL AKHIR (ELITE 50): {mean_acc:.2f}%")
print("="*50)

print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))
    
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Elite Balanced Result\nAcc: {mean_acc:.1f}%')
plt.show()

# --- 6. SAVE ---
if clf_final:
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(clf_final, 'models/stage2b_rf.pkl') 
    joblib.dump(meta_final, 'models/stage2b_meta.pkl')
    print("✅ Model Stage 2B (Elite Version) berhasil disimpan!")