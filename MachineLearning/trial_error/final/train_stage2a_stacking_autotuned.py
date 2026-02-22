import os
import re
import glob
import logging

# Mute Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import joblib 
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['angry', 'happy'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 29: AUTO-TUNED STACKING (RF + SVM META)...")

# --- 1. SETUP & CLEANING DATA ---
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    lyrics_map = pd.Series(df.lyrics.values, index=df.id).to_dict()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Excel Loaded: {len(df)} baris valid.")
except Exception as e:
    print(f"❌ Error loading Excel: {e}")
    exit()

def clean_lyrics_text(text):
    if pd.isna(text) or text == '': return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text) 
    text = re.sub(r'\(.*?\)', ' ', text) 
    garbage = ['lyrics', 'embed', 'contributors', 'translation']
    for w in garbage: text = text.replace(w, '')
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for pid in lyrics_map: lyrics_map[pid] = clean_lyrics_text(lyrics_map[pid])
print("✅ Lirik berhasil dibersihkan.")

print("⏳ Loading Models (YAMNet & RoBERTa)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 2. FEATURE EXTRACTION ---
X_audio_features = []
X_text_scores = [] 
y_labels = []
titles_log = []

print("🧠 Extracting Features (Multi-Source)...")

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
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
        return [s_angry, s_happy] 
    except: return [0.5, 0.5]

def get_id_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

all_audio_files = []
for d in SOURCE_DIRS:
    all_audio_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_audio_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

for file_path in tqdm(all_audio_files, desc="Processing Audio"):
    fid = get_id_from_filename(file_path)
    if fid not in lyrics_map: continue 
    mood = mood_map[fid]
    label = 0 if mood == 'angry' else 1
    
    aud = extract_audio(file_path)
    txt = get_text_scores(lyrics_map[fid])
    
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(label)
        titles_log.append(os.path.basename(file_path))

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) 
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 3. STACKING TRAINING (AUTO-TUNED) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

# Parameter Grids untuk Tuning Otomatis
# Audio Model (RF) Params
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

# Meta Model (SVM) Params
svm_param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto'],
    'svc__kernel': ['rbf', 'linear']
}

print(f"\n🚀 START TUNED STACKING ({FOLDS}-Fold)...")

# Variabel Saving
best_rf_model = None
best_meta_model = None
best_overall_acc = 0

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    print(f"\n🔄 Processing Fold {fold+1}/{FOLDS}...")
    
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # --- STEP 1: TUNE BASE LEARNER (AUDIO RF) ---
    print("   🛠️ Tuning Base Learner (Random Forest)...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=SEED),
        rf_param_grid,
        cv=3, # 3-Fold internal CV
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_aud_tr, y_tr)
    best_rf = rf_grid.best_estimator_
    print(f"      Best RF Params: {rf_grid.best_params_}")
    
    # Generate Probabilities (Stacking Features)
    prob_audio_train = cross_val_predict(best_rf, X_aud_tr, y_tr, cv=3, method='predict_proba')
    prob_audio_test = best_rf.predict_proba(X_aud_ts)
    
    # --- STEP 2: PREPARE META FEATURES ---
    X_meta_train = np.concatenate([prob_audio_train, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # --- STEP 3: TUNE META LEARNER (SVM) ---
    print("   🛠️ Tuning Meta Learner (SVM)...")
    # SVM butuh scaling, jadi pakai Pipeline
    svm_pipeline = make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED))
    
    meta_grid = GridSearchCV(
        svm_pipeline,
        svm_param_grid,
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    meta_grid.fit(X_meta_train, y_tr)
    best_meta = meta_grid.best_estimator_
    print(f"      Best Meta Params: {meta_grid.best_params_}")
    
    # --- STEP 4: PREDICT & EVALUATE ---
    y_pred_fold = best_meta.predict(X_meta_test)
    acc = accuracy_score(y_ts, y_pred_fold)
    
    acc_scores.append(acc)
    print(f"   👉 Fold {fold+1} Acc: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)
    
    # Simpan model jika performa fold ini bagus (untuk referensi)
    if acc > best_overall_acc:
        best_overall_acc = acc
        best_rf_model = best_rf
        best_meta_model = best_meta

# --- 4. REPORT & ANALYSIS ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (AUTO-TUNED STACKING)")
print("="*50)

print(f"🏆 Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")
print("-" * 50)
print(classification_report(y_true_all, y_pred_all, target_names=['angry', 'happy']))

# Plot CM
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Angry','Happy'], yticklabels=['Angry','Happy'])
plt.title(f'Tuned Stacking Result\nMean Acc: {mean_acc:.1f}% ±{std_acc:.1f}%')
plt.show()

# --- 5. SAVING MODEL ---
print("\n💾 Saving Final Tuned Models...")
if best_rf_model and best_meta_model:
    if not os.path.exists('models'): os.makedirs('models')
    
    # Kita harus melatih ulang model terbaik pada SELURUH data agar siap deploy
    print("   Training Final Model on Full Dataset...")
    
    # 1. Train RF on Full Audio Data
    best_rf_model.fit(X_audio_features, y_labels)
    
    # 2. Generate Meta Features on Full Data
    prob_audio_full = best_rf_model.predict_proba(X_audio_features)
    X_meta_full = np.concatenate([prob_audio_full, X_text_scores], axis=1)
    
    # 3. Train Meta on Full Meta Data
    best_meta_model.fit(X_meta_full, y_labels)
    
    joblib.dump(best_rf_model, 'models/stage2a_rf_tuned.pkl')
    joblib.dump(best_meta_model, 'models/stage2a_meta_tuned.pkl')
    print("✅ Model Stage 2A (Tuned) berhasil disimpan!")