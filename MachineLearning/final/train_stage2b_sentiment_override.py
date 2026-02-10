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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- INSTALL VADER LEXICON ---
# VADER adalah alat hitung sentimen yang sangat cepat dan akurat untuk bahasa Inggris
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI EXP 33: SENTIMENT-FIRST APPROACH (SAD vs RELAXED)...")

# --- 1. SETUP DATA ---
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    
    lyrics_map = pd.Series(df.lyrics.values, index=df.id).to_dict()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Loaded: {len(df)} entries.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

def clean_lyrics_text(text):
    if pd.isna(text) or text == '': return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for pid in lyrics_map:
    lyrics_map[pid] = clean_lyrics_text(lyrics_map[pid])

# --- 2. FITUR BARU: SENTIMEN & MAJOR/MINOR ---
sia = SentimentIntensityAnalyzer()

X_features = []
y_labels = []
filenames = []

print("🧠 Extracting Features (Sentiment + Tonality)...")

def extract_features(path, lyrics):
    # 1. AUDIO: Fokus ke Kunci Nada (Major/Minor) & Kecerahan
    try:
        y, sr = librosa.load(path, sr=16000)
        
        # Spectral Centroid (Kecerahan) - Relaxed biasanya lebih 'terang' dari Sad
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        
        # RMS (Energi rata-rata)
        rms_mean = np.mean(librosa.feature.rms(y=y))
        
        # Chroma (Deteksi Nada) -> Statistik sederhananya
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_std = np.std(chroma) # Variasi nada (Sad mungkin lebih monoton/rendah variasinya)
        
    except:
        cent_mean, rms_mean, chroma_std = 0, 0, 0

    # 2. TEXT: VADER Sentiment (KUNCI UTAMA)
    # Compound score: -1 (Sangat Negatif/Sad) s.d +1 (Sangat Positif/Relaxed)
    sentiment = sia.polarity_scores(str(lyrics))
    sent_compound = sentiment['compound'] 
    sent_pos = sentiment['pos']
    sent_neg = sentiment['neg']
    
    # Gabung Fitur: [Vader_Compound, Vader_Pos, Vader_Neg, Audio_Bright, Audio_RMS, Audio_Var]
    return [sent_compound, sent_pos, sent_neg, cent_mean, rms_mean, chroma_std]

def get_id_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

# Kumpulkan File
all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

for file_path in tqdm(all_files):
    fid = get_id_from_filename(file_path)
    if fid not in lyrics_map: continue
    
    mood = mood_map[fid]
    lyric = lyrics_map[fid]
    
    if mood == 'sad': label = 0
    elif mood == 'relaxed': label = 1
    else: continue
    
    feats = extract_features(file_path, lyric)
    X_features.append(feats)
    y_labels.append(label)
    filenames.append(os.path.basename(file_path))

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_features.shape}")
print("   Fitur [0] adalah VADER Score (Kunci Pembeda)")

# --- 3. TRAINING (RANDOM FOREST) ---
# RF sangat bagus menangani fitur campuran (Skor Sentimen + Fitur Audio)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
bad_preds = []

# Model Final
final_model = None

print(f"\n🚀 START TRAINING...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    files_ts = [filenames[i] for i in test_idx]
    
    # Random Forest Sederhana
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    clf.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_ts)
    y_prob = clf.predict_proba(X_ts)
    
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    # Cek yang salah lagi
    for i in range(len(y_ts)):
        if y_ts[i] != y_pred[i]:
            true_cls = 'Relaxed' if y_ts[i]==1 else 'Sad'
            pred_cls = 'Relaxed' if y_pred[i]==1 else 'Sad'
            # Cek skor sentimennya (Fitur ke-0)
            vader_score = X_ts[i][0] 
            
            bad_preds.append({
                'file': files_ts[i],
                'true': true_cls,
                'pred': pred_cls,
                'vader': vader_score # Jika negatif -> Sad, Positif -> Relaxed
            })
            
    final_model = clf

# --- 4. REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (SENTIMENT-FIRST)")
print("="*50)
print(f"🏆 Avg Accuracy : {mean_acc:.2f}%")
print(f"📉 Deviation    : ±{std_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

# --- 5. ANALISIS IMPORTANCE ---
print("\n⚖️ APA YANG DILIHAT MODEL?")
importances = final_model.feature_importances_
feat_names = ['VADER Compound', 'VADER Pos', 'VADER Neg', 'Audio Bright', 'Audio RMS', 'Audio Var']
for name, imp in zip(feat_names, importances):
    print(f"   {name}: {imp:.4f}")

# --- 6. DETEKTIF DATA ---
print("\n🔍 CONTOH KASUS SALAH:")
if len(bad_preds) > 0:
    print("(Perhatikan VADER Score: < -0.05 biasanya Sad, > 0.05 Relaxed)")
    for i in range(min(10, len(bad_preds))):
        item = bad_preds[i]
        print(f"📂 {item['file']}")
        print(f"   Label: {item['true']} | Pred: {item['pred']} | VADER: {item['vader']:.2f}")
        if item['true'] == 'Sad' and item['vader'] > 0.5:
            print("   👉 KASUS: Label Sad, tapi liriknya Positif banget!")
        if item['true'] == 'Relaxed' and item['vader'] < -0.5:
            print("   👉 KASUS: Label Relaxed, tapi liriknya Depresi!")
        print("-" * 30)

# --- 7. SAVE ---
if final_model:
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(final_model, 'models/stage2b_rf_sentiment.pkl')
    print("\n✅ Model disimpan sebagai 'models/stage2b_rf_sentiment.pkl'")