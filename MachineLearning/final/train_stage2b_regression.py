import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import joblib 
from transformers import pipeline

# --- UBAH DARI CLASSIFIER KE REGRESSOR ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
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

print(f"🚀 MEMULAI EXP 35: REGRESSION STACKING (SAD=0.0 vs RELAXED=1.0)...")

# --- 1. SETUP & CLEANING DATA ---
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

print("⏳ Loading Models...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 2. FEATURE EXTRACTION ---
X_audio_features = []
X_text_scores = [] 
y_values = [] # Targetnya Angka Float (0.0 atau 1.0)

print("🧠 Extracting Features (Audio + Text)...")

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        # 1. YAMNet
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        
        # 2. Chroma (PENTING untuk Sad vs Relaxed)
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
        
        # 3. Stats
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        return np.concatenate([yamnet_vec, chroma, [rms, zcr]])
    except: return None

def get_text_scores(lyrics):
    try:
        chunk = str(lyrics)[:512]
        output = nlp_classifier(chunk)[0]
        s = {item['label']: item['score'] for item in output}
        
        # Kita ambil skor mentah Sadness dan Joy/Neutral
        s_sadness = s.get('sadness', 0) + s.get('fear', 0)
        s_relaxed = s.get('neutral', 0) + s.get('joy', 0)
        
        return [s_sadness, s_relaxed]
    except: return [0.5, 0.5]

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
    
    # --- TARGET REGRESI ---
    if mood == 'sad': val = 0.0      # Target 0
    elif mood == 'relaxed': val = 1.0 # Target 1
    else: continue
    
    aud = extract_audio(file_path)
    txt = get_text_scores(lyric)
    
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_values.append(val)

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores)
y_values = np.array(y_values)

print(f"✅ Data Siap: {X_audio_features.shape}")

# --- 3. STACKING REGRESSION TRAINING ---
# Kita butuh label dummy (0/1) hanya untuk StratifiedKFold biar seimbang
y_dummy_labels = [0 if v < 0.5 else 1 for v in y_values]

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_class_all = []
y_pred_score_all = [] # Simpan skor mentah untuk analisis threshold

# Model Final
reg_audio_final = None
meta_reg_final = None
scaler_final = None

print(f"\n🚀 START REGRESSION TRAINING...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_dummy_labels)):
    
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_values[train_idx], y_values[test_idx] # Targetnya Float (0.0/1.0)
    X_txt_tr, X_txt_ts = X_text_scores[train_idx], X_text_scores[test_idx]
    
    # --- LEVEL 1: AUDIO REGRESSOR ---
    # Random Forest Regressor akan mencoba memprediksi angka 0 s.d 1
    reg_audio = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=SEED)
    reg_audio.fit(X_aud_tr, y_tr)
    
    # Prediksi Skor Audio
    score_audio_train = cross_val_predict(reg_audio, X_aud_tr, y_tr, cv=3)
    score_audio_test = reg_audio.predict(X_aud_ts)
    
    # Reshape untuk input Meta
    score_audio_train = score_audio_train.reshape(-1, 1)
    score_audio_test = score_audio_test.reshape(-1, 1)
    
    # --- LEVEL 2: META REGRESSOR (SVR) ---
    # Input: [Skor_Audio, Skor_Teks_Sad, Skor_Teks_Relaxed]
    X_meta_train = np.concatenate([score_audio_train, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([score_audio_test, X_txt_ts], axis=1)
    
    # Scaling (Wajib buat SVR)
    scaler = StandardScaler()
    X_meta_train_sc = scaler.fit_transform(X_meta_train)
    X_meta_test_sc = scaler.transform(X_meta_test)
    
    # SVR (Support Vector Regressor)
    meta_reg = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    meta_reg.fit(X_meta_train_sc, y_tr)
    
    # HASIL PREDIKSI (ANGKA KONTINU, misal: 0.23, 0.88, 0.55)
    y_pred_score = meta_reg.predict(X_meta_test_sc)
    
    # --- CONVERT KE KELAS (THRESHOLD 0.5) ---
    y_pred_class = [0 if s < 0.5 else 1 for s in y_pred_score]
    y_true_class = [0 if s < 0.5 else 1 for s in y_ts]
    
    acc = accuracy_score(y_true_class, y_pred_class)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}% (MSE: {mean_squared_error(y_ts, y_pred_score):.4f})")
    
    y_true_all.extend(y_true_class)
    y_pred_class_all.extend(y_pred_class)
    y_pred_score_all.extend(y_pred_score)
    
    reg_audio_final = reg_audio
    meta_reg_final = meta_reg
    scaler_final = scaler

# --- 4. ANALISIS THRESHOLD TERBAIK ---
print("\n🔍 MENCARI PEMBATAS (THRESHOLD) OPTIMAL...")
best_thr = 0.5
best_acc = 0
thresholds = np.arange(0.3, 0.7, 0.01)

for thr in thresholds:
    temp_pred = [0 if s < thr else 1 for s in y_pred_score_all]
    temp_acc = accuracy_score(y_true_all, temp_pred)
    if temp_acc > best_acc:
        best_acc = temp_acc
        best_thr = thr

print(f"✅ Threshold Terbaik Ditemukan: {best_thr:.2f}")
print(f"   Akurasi dengan Threshold 0.50: {np.mean(acc_scores)*100:.2f}%")
print(f"   Akurasi dengan Threshold {best_thr:.2f}: {best_acc*100:.2f}%")

# --- 5. REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR (REGRESSION)")
print("="*50)
print(classification_report(y_true_all, y_pred_class_all, target_names=['sad', 'relaxed']))

# Histogram Skor Distribusi
plt.figure(figsize=(8, 4))
sns.histplot(y_pred_score_all, bins=20, kde=True, color='purple')
plt.axvline(best_thr, color='red', linestyle='--', label=f'Best Threshold: {best_thr:.2f}')
plt.title("Distribusi Skor Prediksi (0=Sad, 1=Relaxed)")
plt.xlabel("Predicted Score")
plt.legend()
plt.savefig('regression_score_dist.png')
plt.show()

# --- 6. SAVE MODEL ---
print("\n💾 Saving Regression Models...")
if reg_audio_final:
    if not os.path.exists('models'): os.makedirs('models')
    
    # Kita simpan dengan nama _reg agar tidak menimpa yang classifier
    joblib.dump(reg_audio_final, 'models/stage2b_rf_reg.pkl')
    joblib.dump(meta_reg_final, 'models/stage2b_svr_reg.pkl')
    joblib.dump(scaler_final, 'models/stage2b_scaler_reg.pkl')
    
    # Simpan threshold terbaik ke text file kecil (opsional, buat ingetan)
    with open('models/stage2b_threshold.txt', 'w') as f:
        f.write(str(best_thr))
        
    print(f"✅ Model Regression berhasil disimpan! (Threshold: {best_thr:.2f})")