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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIGURATION =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
TARGET_MOODS = ['angry', 'happy', 'sad', 'relaxed']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx' 

# Path Model
MODEL_PATHS = {
    "stage1": "models/stage1_nn.h5",              
    "stage2a_base": "models/stage2a_rf.pkl",   
    "stage2a_meta": "models/stage2a_meta.pkl", 
    "stage2b": "models/stage2b_tuned_final.pkl"     
}

# Config Filtering Stage 2B (HARUS SAMA PERSIS DENGAN TRAINING)
TARGET_COUNT_S2B = 40 
SEED = 43 

print("🚀 MEMUAT EVALUASI GLOBAL (FIXED PADDING & DATA CONSISTENCY)...")

# ================= 1. LOAD EXCEL & VALIDATE IDS =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)]
    
    mood_map = dict(zip(df['id'], df['mood']))
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    for k, v in lyrics_map.items():
        if pd.isna(v): lyrics_map[k] = ""
            
    print(f"📊 Database Excel Loaded: {len(df)} entries.")
except Exception as e:
    print(f"❌ Gagal load excel: {e}")
    exit()

# ================= 2. RE-IDENTIFY ELITE DATA (TRAINING LOGIC REPLICATION) =================
print("\n⚖️ Re-creating Elite Dataset (Logic Copy-Paste from Training)...")

def get_id_smart(path):
    base = os.path.basename(path)
    return base.split('_')[0] if '_' in base else None

def extract_cluster_feat(path):
    # Sama persis dengan training
    y, sr = librosa.load(path, sr=16000)
    return [
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(np.std(librosa.feature.chroma_cens(y=y, sr=sr), axis=1))
    ]

# 1. Kumpulkan Semua File
all_files_training_logic = []
for d in SOURCE_DIRS:
    all_files_training_logic.extend(glob.glob(os.path.join(d, '**/*.wav'), recursive=True))
    all_files_training_logic.extend(glob.glob(os.path.join(d, '**/*.mp3'), recursive=True))

# 2. SORTING (CRITICAL)
all_files_training_logic.sort()

# 3. Filtering & Extraction
X_audio_raw = []
filenames_training = []

print("   Scanning files for Clustering...")
for p in tqdm(all_files_training_logic, desc="Cluster Scan"):
    fid = get_id_smart(p)
    
    if fid not in mood_map: continue
    if mood_map[fid] not in ['sad', 'relaxed']: continue

    try:
        X_audio_raw.append(extract_cluster_feat(p))
        filenames_training.append(p)
    except: pass

# 4. Clustering Process
X_audio_raw = np.array(X_audio_raw)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_audio_raw)

kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=30)
clusters = kmeans.fit_predict(X_scaled)

bright0 = np.mean(X_audio_raw[clusters == 0, 0])
bright1 = np.mean(X_audio_raw[clusters == 1, 0])
sad_cluster, relaxed_cluster = (0, 1) if bright0 < bright1 else (1, 0)

dist = kmeans.transform(X_scaled)

sad_indices = [i for i in range(len(filenames_training)) if clusters[i] == sad_cluster]
rel_indices = [i for i in range(len(filenames_training)) if clusters[i] == relaxed_cluster]

sad_sorted = sorted([(i, dist[i][sad_cluster]) for i in sad_indices], key=lambda x: x[1])[:TARGET_COUNT_S2B]
rel_sorted = sorted([(i, dist[i][relaxed_cluster]) for i in rel_indices], key=lambda x: x[1])[:TARGET_COUNT_S2B]

VALID_LOW_ENERGY_PATHS = set()
for i, _ in sad_sorted + rel_sorted:
    VALID_LOW_ENERGY_PATHS.add(os.path.abspath(filenames_training[i]))

print(f"✅ Filter Siap: {len(VALID_LOW_ENERGY_PATHS)} Elite Sad/Relaxed files terkunci.")

# ================= 3. LOAD MODELS =================
print("\n⏳ Loading Inference Models...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)
model_stage1 = tf.keras.models.load_model(MODEL_PATHS["stage1"])
model_s2a_base = joblib.load(MODEL_PATHS["stage2a_base"])
model_s2a_meta = joblib.load(MODEL_PATHS["stage2a_meta"])
model_stage2b = joblib.load(MODEL_PATHS["stage2b"])

# ================= 4. HELPER EXTRACTORS (FIXED PADDING) =================
# INI PERBAIKAN KRUSIAL DARI ANALISA KAMU

def trim_middle(y, sr, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def extract_s1(y, sr):
    # Padding wajib ada
    if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
        
    if np.max(np.abs(y)) > 0: y_norm = y / np.max(np.abs(y))
    else: y_norm = y
    
    _, emb, _ = yamnet_model(y_norm)
    mean = tf.reduce_mean(emb, axis=0)
    std = tf.math.reduce_std(emb, axis=0)
    max_ = tf.reduce_max(emb, axis=0)
    feat = tf.concat([mean, std, max_], axis=0).numpy()
    
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([feat, [rms, zcr]])

def extract_s2a(y, sr):
    if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
    _, emb, _ = yamnet_model(y) 
    vec = tf.reduce_mean(emb, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([vec, [rms, zcr]])

def extract_s2b(y, sr):
    # 🔥 FIXED: Padding ditambahkan agar SAMA PERSIS dengan Training Stage 2B
    if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
        
    _, emb, _ = yamnet_model(y)
    vec = tf.reduce_mean(emb, axis=0).numpy()
    
    # Chroma & RMS
    chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Pipeline SVM kamu sudah punya StandardScaler, jadi return raw features saja
    return np.concatenate([vec, chroma, [rms]])

def get_text_prob(lyrics):
    if not lyrics or str(lyrics).strip() == "": return [0.5, 0.5]
    text = re.sub(r"[^a-z0-9\s]", '', str(lyrics).lower())
    output = nlp_classifier(text[:512])[0]
    scores = {item['label']: item['score'] for item in output}
    s_hap = scores.get('joy', 0) + scores.get('surprise', 0)
    s_ang = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
    tot = s_hap + s_ang + 1e-9
    return [s_ang/tot, s_hap/tot]

# ================= 5. GLOBAL PREDICTION LOOP =================
y_true = []
y_pred = []

# Gunakan file list yang sudah di-sort dari proses clustering tadi
# Tapi kita butuh juga file angry/happy, jadi load ulang semua dengan aman
all_eval_files = []
for d in SOURCE_DIRS:
    all_eval_files.extend(glob.glob(os.path.join(d, '**/*.wav'), recursive=True))
    all_eval_files.extend(glob.glob(os.path.join(d, '**/*.mp3'), recursive=True))
all_eval_files.sort()

print(f"\n📂 Memulai Evaluasi Global pada {len(all_eval_files)} file...")
skipped_count = 0
processed_count = 0

for path in tqdm(all_eval_files, desc="Evaluasi"):
    fid = get_id_smart(path)
    if fid not in mood_map: continue 
    
    true_label = mood_map[fid] 
    
    # --- LOGIKA FILTERING ---
    if true_label in ['sad', 'relaxed']:
        abs_path = os.path.abspath(path)
        if abs_path not in VALID_LOW_ENERGY_PATHS:
            skipped_count += 1
            continue 
    
    # --- INFERENCE ---
    try:
        y, sr = librosa.load(path, sr=16000)
        # Padding dilakukan di dalam extractor function sekarang
        
        # 1. STAGE 1 (NN)
        y_trim = trim_middle(y, sr)
        feat1 = extract_s1(y_trim, sr).reshape(1, -1)
        p1 = model_stage1.predict(feat1, verbose=0)[0] 
        
        predicted_mood = ""
        stage_info = ""
        
        if p1[0] > p1[1]: # HIGH ENERGY
            # 2. STAGE 2A (Stacking)
            feat2a = extract_s2a(y, sr).reshape(1, -1)
            p_aud = model_s2a_base.predict_proba(feat2a)[0]
            p_txt = np.array(get_text_prob(lyrics_map[fid]))
            
            meta_in = np.concatenate([p_aud, p_txt]).reshape(1, -1)
            final_p = model_s2a_meta.predict_proba(meta_in)[0]
            predicted_mood = "angry" if final_p[0] > final_p[1] else "happy"
            stage_info = "Stage 2A (High)"
            
        else: # LOW ENERGY
            # 3. STAGE 2B (SVM Pipeline)
            feat2b = extract_s2b(y, sr).reshape(1, -1)
            probs = model_stage2b.predict_proba(feat2b)[0]
            predicted_mood = "sad" if probs[0] > probs[1] else "relaxed"
            stage_info = "Stage 2B (Low)"
            
        y_true.append(true_label)
        y_pred.append(predicted_mood)
        processed_count += 1
        
        # --- DEBUG PRINT JIKA SALAH ---
        if predicted_mood != true_label:
            print(f"\n❌ MISCLASSIFIED: {os.path.basename(path)}")
            print(f"   True: {true_label} | Pred: {predicted_mood}")
            print(f"   Route: {stage_info}")
            print(f"   S1 Probs: High={p1[0]:.3f}, Low={p1[1]:.3f}")
        
    except Exception as e:
        print(f"Err {os.path.basename(path)}: {e}")

# ================= 6. REPORTING =================
print("\n" + "="*60)
print("📊 FIXED GLOBAL SYSTEM PERFORMANCE")
print("="*60)
print(f"Dievaluasi (Clean Data) : {processed_count}")
print(f"Dibuang (Noise Low)     : {skipped_count}")
print("-" * 60)

global_acc = accuracy_score(y_true, y_pred) * 100
print(f"🏆 ADJUSTED GLOBAL ACCURACY: {global_acc:.2f}%")
print("-" * 60)
print(classification_report(y_true, y_pred, target_names=TARGET_MOODS))

cm = confusion_matrix(y_true, y_pred, labels=TARGET_MOODS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=TARGET_MOODS, yticklabels=TARGET_MOODS)
plt.title(f'Global CM (Fixed Padding)\nAcc: {global_acc:.2f}%')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

print("\n✅ Evaluasi Selesai! Jika akurasi Stage 2B masih rendah, cek debug logs di atas.")