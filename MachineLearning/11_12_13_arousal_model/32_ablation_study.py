import os
import logging
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- CONFIG LOCKED ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics.csv'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5 
SEED = 43 # Seed SAMA dengan Exp 31 biar adil
N_ESTIMATORS = 200

print(f"🚀 MEMULAI ABLATION STUDY (PEMBUKTIAN KONTRIBUSI)...")

# --- DATA LOADING & MODEL ---
# (Bagian ini sama persis dengan Exp 31, disingkat biar fokus logika)
if not os.path.exists(LYRICS_PATH): exit()
df = pd.read_csv(LYRICS_PATH, sep=';')
if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',')
df.columns = df.columns.str.strip().str.lower()
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- FEATURE EXTRACTION (TERPISAH) ---
X_audio = []
X_text = []
y_labels = []

files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])
lyrics_sad = df[df['mood']=='sad']['lyrics'].tolist()
lyrics_relaxed = df[df['mood']=='relaxed']['lyrics'].tolist()

def extract_modality(audio_path, lyrics_text):
    # AUDIO
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        y_harm = librosa.effects.harmonic(y)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr), axis=1)
        aud_feat = np.concatenate([yamnet_vec, tonnetz])
    except: return None, None

    # TEXT
    try:
        if pd.isna(lyrics_text) or len(str(lyrics_text)) < 2:
            txt_feat = [0.0]*6
        else:
            output = nlp_classifier(str(lyrics_text))[0]
            scores = {item['label']: item['score'] for item in output}
            txt_feat = [scores.get('joy',0), scores.get('neutral',0), scores.get('surprise',0),
                        scores.get('sadness',0), scores.get('fear',0), scores.get('anger',0)]
    except: return None, None
    
    return aud_feat, txt_feat

print("🧠 Extracting Separate Modalities...")
for i in tqdm(range(min(len(files_sad), len(lyrics_sad))), desc="Processing"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    a, t = extract_modality(path, lyrics_sad[i])
    if a is not None: X_audio.append(a); X_text.append(t); y_labels.append(0)

for i in tqdm(range(min(len(files_relaxed), len(lyrics_relaxed))), desc="Processing"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    a, t = extract_modality(path, lyrics_relaxed[i])
    if a is not None: X_audio.append(a); X_text.append(t); y_labels.append(1)

X_audio = np.array(X_audio)
X_text = np.array(X_text)
y_labels = np.array(y_labels)

# --- RUN ABLATION ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

results = {'Audio Only': [], 'Hybrid (Final)': []}

print(f"\n🥊 BANDINGKAN: AUDIO ONLY vs HYBRID...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio, y_labels)):
    # 1. AUDIO ONLY RUN
    clf_aud = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, class_weight='balanced')
    clf_aud.fit(X_audio[train_idx], y_labels[train_idx])
    acc_aud = accuracy_score(y_labels[test_idx], clf_aud.predict(X_audio[test_idx]))
    results['Audio Only'].append(acc_aud)
    
    # 2. HYBRID RUN (Concatenate)
    X_hybrid_train = np.concatenate([X_audio[train_idx], X_text[train_idx]], axis=1)
    X_hybrid_test = np.concatenate([X_audio[test_idx], X_text[test_idx]], axis=1)
    
    clf_hyb = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, class_weight='balanced')
    clf_hyb.fit(X_hybrid_train, y_labels[train_idx])
    acc_hyb = accuracy_score(y_labels[test_idx], clf_hyb.predict(X_hybrid_test))
    results['Hybrid (Final)'].append(acc_hyb)
    
    print(f"   Fold {fold+1}: Audio={acc_aud*100:.0f}% vs Hybrid={acc_hyb*100:.0f}%")

# --- FINAL VERDICT ---
print("\n" + "="*50)
print("🏆 ABLATION STUDY RESULTS")
print("="*50)
avg_aud = np.mean(results['Audio Only']) * 100
avg_hyb = np.mean(results['Hybrid (Final)']) * 100

print(f"🎵 Audio Only Accuracy : {avg_aud:.2f}%")
print(f"🔗 Hybrid Accuracy     : {avg_hyb:.2f}%")
print(f"📈 Improvement         : +{avg_hyb - avg_aud:.2f}%")

if avg_hyb >= avg_aud:
    print("\n✅ VALIDASI BERHASIL: Penambahan Teks meningkatkan/menjaga akurasi.")
    print("   (Meskipun kontribusi fitur kecil, ia menstabilkan model).")
else:
    print("\n⚠️ WARNING: Teks justru menurunkan akurasi (Noise).")