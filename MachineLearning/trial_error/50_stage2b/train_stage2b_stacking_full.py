import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
SEED = 43

print("🚀 TRAINING STAGE 2B: STACKING (FULL DATASET)")
print("   Strategy: Use ALL available Sad/Relaxed songs with lyrics.")

# ================= 1. PREPARE DATA & LYRICS =================
try:
    df = pd.read_excel(LYRICS_PATH)
    # Bersihkan ID
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)]
    
    # Map Lirik & Mood
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    mood_map = dict(zip(df['id'], df['mood']))
    
    # Bersihkan NaN lirik
    for k, v in lyrics_map.items():
        if pd.isna(v): lyrics_map[k] = "" 
            
    print(f"📊 Excel Database Loaded: {len(df)} valid entries.")
except Exception as e:
    print(f"❌ Error loading Excel: {e}")
    exit()

# ================= 2. COLLECT ALL FILES (NO LIMIT) =================
def get_id_smart(path):
    base = os.path.basename(path)
    return base.split('_')[0] if '_' in base else None

print("\n🔍 Scanning ALL Files in Source Dirs...")
raw_files = []
for d in SOURCE_DIRS:
    # Ambil semua file di folder sad & relaxed
    raw_files.extend(glob.glob(os.path.join(d, 'sad', '*.wav')) + glob.glob(os.path.join(d, 'sad', '*.mp3')))
    raw_files.extend(glob.glob(os.path.join(d, 'relaxed', '*.wav')) + glob.glob(os.path.join(d, 'relaxed', '*.mp3')))

# Sort agar urutan konsisten
raw_files.sort()

valid_files = []
valid_labels = [] # 0=Sad, 1=Relaxed
valid_lyrics = []

count_sad = 0
count_rel = 0

print("   Filtering Valid Files (Must have Lyrics)...")
for p in tqdm(raw_files):
    fid = get_id_smart(p)
    
    # Syarat 1: ID harus ada di Excel (karena kita butuh liriknya)
    if fid not in mood_map: continue
    
    label_str = mood_map[fid]
    
    # Syarat 2: Label harus sad/relaxed (Double check)
    if label_str == 'sad':
        lbl = 0
        count_sad += 1
    elif label_str == 'relaxed':
        lbl = 1
        count_rel += 1
    else:
        continue
    
    # Masukkan ke dataset (TANPA BATASAN JUMLAH)
    valid_files.append(p)
    valid_labels.append(lbl)
    valid_lyrics.append(lyrics_map[fid])

print(f"\n✅ FINAL DATASET: {len(valid_files)} Total Files")
print(f"   - Sad    : {count_sad}")
print(f"   - Relaxed: {count_rel}")

# ================= 3. FEATURE EXTRACTION =================
print("\n⏳ Loading Feature Extractors...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

# A. Audio Features (YAMNet + Chroma + RMS)
def get_audio_feat(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        # Padding Konsisten
        if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
            
        _, emb, _ = yamnet_model(y)
        vec = tf.reduce_mean(emb, axis=0).numpy()
        chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
        rms = np.mean(librosa.feature.rms(y=y))
        
        return np.concatenate([vec, chroma, [rms]])
    except:
        return np.zeros(1024 + 12 + 1) # Fallback jika error (jarang terjadi)

# B. Text Probabilities
def get_text_prob(lyrics):
    if not lyrics or str(lyrics).strip() == "": return [0.5, 0.5]
    try:
        text = re.sub(r"[^a-z0-9\s]", '', str(lyrics).lower())
        output = nlp_classifier(text[:512])[0]
        scores = {item['label']: item['score'] for item in output}
        
        # Mapping Emosi Teks
        s_sad = scores.get('sadness', 0) + scores.get('fear', 0) + scores.get('anger', 0) + scores.get('disgust', 0)
        s_rel = scores.get('joy', 0) + scores.get('neutral', 0) + scores.get('surprise', 0)
        
        total = s_sad + s_rel + 1e-9
        return [s_sad/total, s_rel/total]
    except: return [0.5, 0.5]

print("   Extracting Features (This may take a while for all files)...")
X_audio = np.array([get_audio_feat(f) for f in tqdm(valid_files, desc="Audio")])
X_text = np.array([get_text_prob(l) for l in tqdm(valid_lyrics, desc="Text")])
y = np.array(valid_labels)

# ================= 4. STACKING TRAINING =================
print("\n🚀 Training Stacking Ensemble...")

# 1. Base Model Audio (Random Forest)
# Kita naikkan estimators karena data lebih banyak
rf_audio = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)

# Cross Val Predict untuk Meta Features (Mencegah Overfitting)
print("   Generating Meta-Features (CV Predict)...")
cv_probs_audio = cross_val_predict(rf_audio, X_audio, y, cv=5, method='predict_proba')

# Gabungkan Prob Audio + Prob Text
# X_meta = [Prob_Sad_Aud, Prob_Rel_Aud, Prob_Sad_Txt, Prob_Rel_Txt]
X_meta = np.concatenate([cv_probs_audio, X_text], axis=1)

# 2. Meta Learner (Logistic Regression)
# C=1.0 default sudah bagus, class_weight='balanced' untuk jaga-jaga kalau jumlah data timpang
meta_clf = LogisticRegression(random_state=SEED, class_weight='balanced') 
meta_preds = cross_val_predict(meta_clf, X_meta, y, cv=5)

# Report
acc = accuracy_score(y, meta_preds) * 100
print("\n" + "="*50)
print(f"🏆 STACKING ACCURACY (FULL DATA): {acc:.2f}%")
print("="*50)
print(classification_report(y, meta_preds, target_names=['Sad', 'Relaxed']))

# Confusion Matrix
cm = confusion_matrix(y, meta_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad', 'Relaxed'], yticklabels=['Sad', 'Relaxed'])
plt.title(f'Stacking Full Data\nAcc: {acc:.2f}%')
plt.show()

# ================= 5. SAVE MODELS =================
print("\n💾 Saving Models...")

# Fit semua model dengan full data
rf_audio.fit(X_audio, y)
meta_clf.fit(X_meta, y)

if not os.path.exists('models'): os.makedirs('models')

# Simpan dengan nama yg konsisten untuk stage 2B Stacking
joblib.dump(rf_audio, 'models/stage2b_base_rf.pkl')
joblib.dump(meta_clf, 'models/stage2b_meta.pkl')

# Update Info
with open('models/stage2b_info.txt', 'w') as f:
    f.write(f"Model: Stacking (RF Audio + RoBERTa Text) - FULL DATA\n")
    f.write(f"Accuracy: {acc:.2f}%\n")
    f.write(f"Data Count: Sad={count_sad}, Relaxed={count_rel}\n")
    f.write("Note: Label 0=Sad, Label 1=Relaxed\n")

print("✅ Model Stage 2B (Full Data) Saved!")