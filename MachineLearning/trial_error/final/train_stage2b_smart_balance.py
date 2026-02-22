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
MIN_SAMPLES = 25 # Target minimal per kelas
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI EXP 38: SMART BALANCING (MIN {MIN_SAMPLES} SAMPLES/CLASS)...")

# --- 1. SETUP DATA ---
try:
    df = pd.read_excel(LYRICS_PATH)
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

# --- 2. SMART CLEANING (PRUNING) ---
# Kita akan menilai "Error Score" tiap data
# Error Score tinggi = Label Excel sangat bertentangan dengan VADER

data_candidates = []

print("⚖️ Menghitung Skor Inkonsistensi...")
for fid, mood in mood_map.items():
    lyric = clean_lyrics(lyrics_map.get(fid, ""))
    vader = sia.polarity_scores(lyric)['compound']
    
    # Hitung seberapa "salah" labelnya
    # Jika Label Relaxed (Target 1.0) tapi Vader -0.8 -> Error besar (1.8)
    # Jika Label Sad (Target -1.0) tapi Vader 0.8 -> Error besar (1.8)
    
    if mood == 'relaxed':
        # Relaxed harusnya positif. Jika negatif, errornya besar.
        # Kita ingin Vader mendekati 1. 
        # Error = Jarak dari 1.0 (tapi kita pakai threshold simple aja)
        inconsistency = -vader # Semakin negatif vader, semakin tinggi inkonsistensi
    else: # mood == 'sad'
        # Sad harusnya negatif. Jika positif, errornya besar.
        inconsistency = vader # Semakin positif vader, semakin tinggi inkonsistensi
        
    data_candidates.append({
        'id': fid,
        'mood': mood,
        'vader': vader,
        'inconsistency': inconsistency
    })

# Pisahkan per kelas
sad_candidates = [d for d in data_candidates if d['mood'] == 'sad']
relaxed_candidates = [d for d in data_candidates if d['mood'] == 'relaxed']

# Urutkan dari yang PALING NGACAO (Inkonsistensi Tinggi) ke Paling Benar
sad_candidates.sort(key=lambda x: x['inconsistency'], reverse=True)
relaxed_candidates.sort(key=lambda x: x['inconsistency'], reverse=True)

print(f"   Kandidat Awal: Sad={len(sad_candidates)}, Relaxed={len(relaxed_candidates)}")

# --- LOGIKA SMART KEEPING ---
# Kita hanya membuang jika jumlah sisa masih > MIN_SAMPLES
# Dan kita hanya membuang yang inkonsistensinya POSITIF (artinya memang salah arah)

final_whitelist = set()

def prune_candidates(candidates, class_name):
    kept = []
    dropped = 0
    
    # Kita butuh setidaknya MIN_SAMPLES terbaik
    # Jadi kita ambil MIN_SAMPLES teratas (yang paling konsisten/inkonsistensi rendah)
    # List sudah diurutkan dari JELEK ke BAGUS. Jadi kita harus ambil dari BELAKANG.
    
    # Sort ulang dari BAGUS (Inkonsistensi Rendah/Negatif) ke JELEK
    candidates.sort(key=lambda x: x['inconsistency']) 
    
    # Ambil data yang "Bagus" (Inkonsistensi < 0.1)
    # TAPI pastikan minimal ambil MIN_SAMPLES
    
    for i, item in enumerate(candidates):
        # Jika kita sudah punya cukup data, kita bisa mulai filter yang jelek
        if len(candidates) - i > MIN_SAMPLES:
            # Filter ketat: Jika inkonsistensi > 0.15 (Label salah parah), buang
            if item['inconsistency'] > 0.15:
                dropped += 1
                continue
            else:
                kept.append(item)
        else:
            # Sisa data sudah mepet batas minimal, TERPAKSA SIMPAN walau jelek
            kept.append(item)
            
    print(f"   Kelas {class_name}: Disimpan {len(kept)} (Dibuang {dropped} Terburuk)")
    return [x['id'] for x in kept]

valid_sad_ids = prune_candidates(sad_candidates, "Sad")
valid_relaxed_ids = prune_candidates(relaxed_candidates, "Relaxed")

valid_ids = set(valid_sad_ids + valid_relaxed_ids)
print(f"✅ Total Data Training Final: {len(valid_ids)}")

# --- 3. MATCHING FILE & EXTRACT ---
print("⏳ Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

X_features = []
y_labels = []

# Fungsi Extract ID yang Robust (V3)
def get_id_smart(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
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

all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

print("⚙️ Memulai Ekstraksi...")
for path in tqdm(all_files):
    file_id = get_id_smart(path)
    if file_id is None: continue 
    
    # Logic Matching ID (String/Int)
    final_id = None
    if file_id in valid_ids: final_id = file_id
    else:
        try:
            if str(int(file_id)) in valid_ids: final_id = str(int(file_id))
        except: pass
            
    if final_id is None: continue

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

print(f"📊 Distribusi Training: Sad={np.sum(y_labels==0)}, Relaxed={np.sum(y_labels==1)}")

# --- 4. TRAINING (STACKING) ---
print(f"\n🚀 START TRAINING...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

clf_final = None
meta_final = None

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    # Base RF (Kita naikkan complexity dikit karena data lebih banyak/kotor)
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=SEED)
    clf.fit(X_tr, y_tr)
    
    try:
        prob_tr = cross_val_predict(clf, X_tr, y_tr, cv=3, method='predict_proba')
    except:
        prob_tr = clf.predict_proba(X_tr)
        
    prob_ts = clf.predict_proba(X_ts)
    
    # Meta LR
    meta = LogisticRegression(C=0.5, class_weight='balanced', random_state=SEED)
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
print(f"🏆 HASIL AKHIR (SMART BALANCE): {mean_acc:.2f}%")
print("="*50)

print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))
    
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Balanced Result\nAcc: {mean_acc:.1f}%')
plt.show()

# --- 6. SAVE ---
if clf_final:
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(clf_final, 'models/stage2b_rf.pkl') 
    joblib.dump(meta_final, 'models/stage2b_meta.pkl')
    print("✅ Model Stage 2B (Smart Balance) berhasil disimpan!")