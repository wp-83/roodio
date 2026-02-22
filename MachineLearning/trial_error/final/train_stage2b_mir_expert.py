import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import librosa
import joblib 
import tensorflow_hub as hub # Masih dipakai hanya untuk Text (RoBERTa)
from transformers import pipeline

# Ganti Deep Learning dengan Gradient Boosting (Lebih jago untuk data tabular/fitur manual)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI EXP 32: MIR FEATURE ENGINEERING (SAD vs RELAXED)...")

# --- 1. SETUP DATA (SAMA SEPERTI SEBELUMNYA) ---
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

print("⏳ Loading RoBERTa (Text Only)...")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 2. ADVANCED MIR FEATURE EXTRACTION (PENGGANTI YAMNET) ---
X_features = [] # Audio + Text gabung langsung
y_labels = []
filenames = []

print("🧠 Extracting MIR Features (Harmoni, Timbre, Rhythm)...")

def extract_mir_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        
        # 1. RMS Energy (Kekerasan Suara)
        # Sad biasanya dinamis, Relaxed stabil
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # 2. Spectral Centroid (Kecerahan)
        # Relaxed biasanya lebih "warm" (rendah), Sad bisa "dark" atau "piercing"
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)
        
        # 3. Spectral Flatness (Noise vs Tone)
        # Relaxed lebih tonal (musik), Sad bisa noisy (nafas/vokal fry)
        flat = librosa.feature.spectral_flatness(y=y)
        flat_mean = np.mean(flat)
        
        # 4. Zero Crossing Rate (Kekasaran)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        
        # 5. Chroma Features (Harmoni/Kunci Nada) - PENTING!
        # Mengukur apakah lagu Major atau Minor secara statistik
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1) # Rata-rata 12 nada
        chroma_std = np.std(chroma, axis=1)   # Variansi 12 nada
        
        # 6. Spectral Contrast (Kekayaan Suara)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        # 7. Harmonic vs Percussive
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_mean = np.mean(y_harm)
        perc_mean = np.mean(y_perc) # Relaxed biasanya low percussive
        
        # Gabungkan semua jadi satu vector
        features = np.concatenate([
            [rms_mean, rms_std, cent_mean, cent_std, flat_mean, zcr_mean, harm_mean, perc_mean],
            chroma_mean, 
            chroma_std,
            contrast_mean
        ])
        return features
    except Exception as e:
        return None

def get_text_scores_full(lyrics):
    try:
        chunk = str(lyrics)[:512]
        output = nlp_classifier(chunk)[0]
        scores = {item['label']: item['score'] for item in output}
        # Urutan: anger, disgust, fear, joy, neutral, sadness, surprise
        sorted_keys = sorted(scores.keys()) 
        return [scores[k] for k in sorted_keys]
    except: return [0.0] * 7

def get_id_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

# Loop Files
all_audio_files = []
for d in SOURCE_DIRS:
    all_audio_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_audio_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

for file_path in tqdm(all_audio_files):
    fid = get_id_from_filename(file_path)
    if fid not in lyrics_map: continue
    
    mood = mood_map[fid]
    lyric = lyrics_map[fid]
    
    if mood == 'sad': label = 0
    elif mood == 'relaxed': label = 1
    else: continue
    
    # Extract Audio (MIR)
    aud_feat = extract_mir_features(file_path)
    # Extract Text (RoBERTa)
    txt_feat = get_text_scores_full(lyric)
    
    if aud_feat is not None:
        # GABUNGKAN FITUR DI AWAL (Early Fusion)
        # Karena kita pakai Gradient Boosting, dia pintar memilih fitur mana yang penting
        combined_feat = np.concatenate([aud_feat, txt_feat])
        
        X_features.append(combined_feat)
        y_labels.append(label)
        filenames.append(os.path.basename(file_path))

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_features.shape} (Fitur Manual + Teks)")

# --- 3. TRAINING: GRADIENT BOOSTING (XGBoost Style) ---
# Gradient Boosting lebih tahan banting daripada SVM untuk fitur campuran (Audio+Teks)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
bad_predictions = [] # Untuk Human-in-the-Loop

# Model Final
final_model = None
final_scaler = None

print(f"\n🚀 START GRADIENT BOOSTING TRAINING...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    files_ts = [filenames[i] for i in test_idx]
    
    # Scaling (Penting untuk fitur audio yang range-nya beda-beda)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_ts_sc = scaler.transform(X_ts)
    
    # Classifier: Gradient Boosting
    clf = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1,
        max_depth=4, 
        random_state=SEED
    )
    clf.fit(X_tr_sc, y_tr)
    
    y_pred = clf.predict(X_ts_sc)
    y_prob = clf.predict_proba(X_ts_sc) # [Prob_Sad, Prob_Relaxed]
    
    acc = accuracy_score(y_ts, y_pred)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    # --- DETEKSI DATA MENYIMPANG (SUSPECT DETECTION) ---
    for i in range(len(y_ts)):
        # Jika Salah Prediksi DAN Pede Banget (>70%)
        confidence = max(y_prob[i])
        if y_ts[i] != y_pred[i] and confidence > 0.70:
            true_cls = 'Relaxed' if y_ts[i]==1 else 'Sad'
            pred_cls = 'Relaxed' if y_pred[i]==1 else 'Sad'
            bad_predictions.append({
                'file': files_ts[i],
                'true': true_cls,
                'pred': pred_cls,
                'conf': f"{confidence*100:.1f}%"
            })

    final_model = clf
    final_scaler = scaler

# --- 4. REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (MIR EXPERT)")
print("="*50)
print(f"🏆 Avg Accuracy : {mean_acc:.2f}%")
print(f"📉 Deviation    : ±{std_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'MIR + Gradient Boosting\nAcc: {mean_acc:.1f}%')
plt.show()

# --- 5. HUMAN-IN-THE-LOOP REPORT (PENTING!) ---
print("\n🔍 DAFTAR FILE 'SUSPECT' (MODEL PEDE TAPI SALAH)")
print("Saran: Cek manual file-file ini. Apakah label di Excel salah?")
print("-" * 60)
if len(bad_predictions) > 0:
    for item in bad_predictions:
        print(f"📂 {item['file']}")
        print(f"   Label Asli: {item['true']} | Model Bilang: {item['pred']} (Yakin {item['conf']})")
        print("-" * 30)
else:
    print("✅ Tidak ada suspect (atau model tidak pede saat salah).")

# --- 6. SAVE MODEL ---
print("\n💾 Saving Models...")
if final_model:
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(final_model, 'models/stage2b_gb.pkl') # Ganti nama jadi GB (Gradient Boosting)
    joblib.dump(final_scaler, 'models/stage2b_mir_scaler.pkl') 
    print("✅ Model Stage 2B (MIR + GB) berhasil disimpan!")