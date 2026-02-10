import os
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2']  # Folder data asli kamu
TARGET_MOODS = ['angry', 'happy', 'sad', 'relaxed']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx' 

# Path Model
MODEL_PATHS = {
    "stage1": "models/stage1_nn.h5",              
    "stage2a_base": "models/stage2a_rf.pkl",   
    "stage2a_meta": "models/stage2a_meta.pkl", 
    "stage2b": "models/stage2b_tuned_final.pkl"     
}

print("🚀 MEMULAI GLOBAL EVALUATION (FULL SYSTEM)...")

# --- 1. LOAD DATA & LYRICS ---
# Kita butuh label asli untuk cek kebenaran
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    # Filter hanya 4 mood target
    df = df[df['mood'].isin(TARGET_MOODS)]
    
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    # Jika lirik kosong, ganti string kosong
    for k, v in lyrics_map.items():
        if pd.isna(v): lyrics_map[k] = ""
            
    print(f"📊 Database Lirik Loaded: {len(df)}")
except:
    print("❌ Gagal load excel lirik. Pastikan path benar.")
    exit()

# --- 2. LOAD MODELS (SAMA SEPERTI INFERENCE) ---
print("⏳ Loading Models...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)
model_stage1 = tf.keras.models.load_model(MODEL_PATHS["stage1"])
model_s2a_base = joblib.load(MODEL_PATHS["stage2a_base"])
model_s2a_meta = joblib.load(MODEL_PATHS["stage2a_meta"])
model_stage2b = joblib.load(MODEL_PATHS["stage2b"])

# --- 3. HELPER FUNCTIONS (COPY DARI INFERENCE) ---
# ... (Untuk menghemat tempat, asumsikan fungsi extract_features sama persis dengan main_inference_stacking.py)
# ... (Saya tulis ulang versi ringkasnya disini agar script ini jalan mandiri)

def trim_middle(y, sr, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def extract_s1(y, sr):
    if np.max(np.abs(y)) > 0: y_norm = y / np.max(np.abs(y))
    else: y_norm = y
    _, embeddings, _ = yamnet_model(y_norm)
    mean = tf.reduce_mean(embeddings, axis=0)
    std = tf.math.reduce_std(embeddings, axis=0)
    max_ = tf.reduce_max(embeddings, axis=0)
    emb = tf.concat([mean, std, max_], axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([emb, [rms, zcr]])

def extract_s2a(y, sr):
    _, embeddings, _ = yamnet_model(y) 
    vec = tf.reduce_mean(embeddings, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([vec, [rms, zcr]])

def extract_s2b(y, sr):
    _, embeddings, _ = yamnet_model(y)
    vec = tf.reduce_mean(embeddings, axis=0).numpy()
    chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    return np.concatenate([vec, chroma, [rms]])

def get_text_prob(lyrics):
    if not lyrics or str(lyrics).strip() == "": return [0.5, 0.5]
    text = str(lyrics).lower()
    text = re.sub(r"[^a-z0-9\s]", '', text)
    output = nlp_classifier(text[:512])[0]
    scores = {item['label']: item['score'] for item in output}
    s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
    s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
    total = s_happy + s_angry + 1e-9
    return [s_angry/total, s_happy/total]

def get_id(path):
    base = os.path.basename(path)
    return base.split('_')[0].strip() if '_' in base else None

# --- 4. PREDICTION LOOP ---
y_true = []
y_pred = []

# Kumpulkan semua file audio
all_files = []
for d in SOURCE_DIRS:
    for m in TARGET_MOODS:
        path = os.path.join(d, m)
        all_files.extend(glob.glob(os.path.join(path, "*.wav")))
        all_files.extend(glob.glob(os.path.join(path, "*.mp3")))

print(f"📂 Total File Ditemukan: {len(all_files)}")
print("   Memulai Prediksi Massal...")

for path in tqdm(all_files):
    fid = get_id(path)
    # Skip jika tidak punya label/lirik di excel (agar validasi fair)
    if fid not in lyrics_map: continue 
    
    # Label Asli (Ground Truth)
    # Kita ambil dari nama folder parentnya (misal: data/raw/angry/song.wav -> angry)
    true_label = os.path.basename(os.path.dirname(path)).lower()
    if true_label not in TARGET_MOODS: continue

    # --- INFERENCE PROCESS ---
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        # 1. STAGE 1 (NN)
        y_trim = trim_middle(y, sr)
        feat1 = extract_s1(y_trim, sr).reshape(1, -1)
        p1 = model_stage1.predict(feat1, verbose=0)[0] # [High, Low] - Mapping Baru
        
        predicted_mood = ""
        
        if p1[0] > p1[1]: # HIGH ENERGY
            # 2. STAGE 2A (Stacking)
            feat2a = extract_s2a(y, sr).reshape(1, -1)
            p_aud = model_s2a_base.predict_proba(feat2a)[0]
            p_txt = np.array(get_text_prob(lyrics_map[fid]))
            
            # Meta Input
            meta_in = np.concatenate([p_aud, p_txt]).reshape(1, -1)
            final_p = model_s2a_meta.predict_proba(meta_in)[0]
            
            predicted_mood = "angry" if final_p[0] > final_p[1] else "happy"
            
        else: # LOW ENERGY
            # 3. STAGE 2B (SVM)
            feat2b = extract_s2b(y, sr).reshape(1, -1)
            probs = model_stage2b.predict_proba(feat2b)[0]
            predicted_mood = "sad" if probs[0] > probs[1] else "relaxed"
            
        y_true.append(true_label)
        y_pred.append(predicted_mood)
        
    except Exception as e:
        print(f"Err {os.path.basename(path)}: {e}")

# --- 5. REPORTING ---
print("\n" + "="*60)
print("📊 GLOBAL SYSTEM PERFORMANCE")
print("="*60)

global_acc = accuracy_score(y_true, y_pred) * 100
print(f"🏆 GLOBAL ACCURACY: {global_acc:.2f}%")
print("-" * 60)
print(classification_report(y_true, y_pred, target_names=TARGET_MOODS))

# Plot 4x4 Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=TARGET_MOODS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=TARGET_MOODS, yticklabels=TARGET_MOODS)
plt.title(f'Global System Confusion Matrix\nAccuracy: {global_acc:.2f}%')
plt.xlabel('Predicted Mood')
plt.ylabel('True Mood')
plt.savefig('global_confusion_matrix.png')
plt.show()

print("\n✅ Evaluasi Selesai! Simpan gambar 'global_confusion_matrix.png' untuk presentasi.")