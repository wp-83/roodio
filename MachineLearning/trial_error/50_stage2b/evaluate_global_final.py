import os
import re
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

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIGURATION =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['angry', 'happy', 'sad', 'relaxed']

# PATH MODEL (KITA PAKAI YANG TERBAIK)
MODEL_PATHS = {
    "stage1": "models/stage1_nn.h5",              
    "stage2a_base": "models/stage2a_rf.pkl",   
    "stage2a_meta": "models/stage2a_meta.pkl", 
    "stage2b": "models/stage2b_purified_rf.pkl" # Model Juara 87%
}

print("🚀 GLOBAL SYSTEM EVALUATION")
print("   Architecture: Hierarchical Multimodal (NN -> Stacking -> Purified RF)")

# ================= 1. LOAD DATA & MODELS =================
try:
    print("\n⏳ Loading Lyrics Database...")
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    
    print("⏳ Loading Models & Extractors...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)
    
    # Load Models
    model_stage1 = tf.keras.models.load_model(MODEL_PATHS["stage1"])
    model_s2a_base = joblib.load(MODEL_PATHS["stage2a_base"])
    model_s2a_meta = joblib.load(MODEL_PATHS["stage2a_meta"])
    model_s2b = joblib.load(MODEL_PATHS["stage2b"]) # The 87% Champion
    
    print("✅ System Loaded Successfully.")

except Exception as e:
    print(f"❌ Error Setup: {e}")
    exit()

# ================= 2. FEATURE EXTRACTORS =================

def trim_middle(y, sr, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# --- Extractor Stage 1 (NN) ---
def extract_s1(y, sr):
    if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
    y_norm = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    
    _, emb, _ = yamnet_model(y_norm)
    mean = tf.reduce_mean(emb, axis=0)
    std = tf.math.reduce_std(emb, axis=0)
    max_ = tf.reduce_max(emb, axis=0)
    feat = tf.concat([mean, std, max_], axis=0).numpy()
    
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([feat, [rms, zcr]])

# --- Extractor Stage 2A (Stacking) ---
def extract_s2a_audio(y, sr):
    if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
    _, emb, _ = yamnet_model(y) 
    vec = tf.reduce_mean(emb, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([vec, [rms, zcr]])

def extract_s2a_text(lyrics):
    if not lyrics: return [0.5, 0.5]
    text = re.sub(r"[^a-z0-9\s]", '', str(lyrics).lower())[:512]
    try:
        res = nlp_classifier(text)[0]
        s = {item['label']: item['score'] for item in res}
        s_hap = s.get('joy', 0) + s.get('surprise', 0)
        s_ang = s.get('anger', 0) + s.get('disgust', 0) + s.get('fear', 0)
        tot = s_hap + s_ang + 1e-9
        return [s_ang/tot, s_hap/tot]
    except: return [0.5, 0.5]

# --- Extractor Stage 2B (Purified RF) ---
def extract_s2b(y, sr, lyrics):
    # 1. Audio
    if len(y) < 16000: y = np.pad(y, (0, max(0, 16000 - len(y))))
    _, emb, _ = yamnet_model(y)
    aud_vec = tf.reduce_mean(emb, axis=0).numpy()
    chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    feat_aud = np.concatenate([aud_vec, chroma, [rms]])
    
    # 2. Text (Sadness vs Joy/Neutral)
    text = re.sub(r"[^a-z0-9\s]", '', str(lyrics).lower())[:512]
    s_sad, s_rel = 0, 0
    if text:
        try:
            res = nlp_classifier(text)[0]
            s = {item['label']: item['score'] for item in res}
            s_sad = s.get('sadness', 0) + s.get('fear', 0)
            s_rel = s.get('joy', 0) + s.get('neutral', 0)
        except: pass
    
    return np.concatenate([feat_aud, [s_sad, s_rel]])

# ================= 3. MAIN PREDICTION LOOP =================
y_true = []
y_pred = []

# Collect Files
all_files = []
for d in SOURCE_DIRS:
    for m in TARGET_MOODS:
        fs = glob.glob(os.path.join(d, m, '*.wav')) + glob.glob(os.path.join(d, m, '*.mp3'))
        all_files.extend(fs)

print(f"\n📂 Memproses {len(all_files)} file audio...")

for path in tqdm(all_files):
    fid = os.path.basename(path).split('_')[0]
    
    # Skip jika tidak punya lirik (Karena Model Stage 2A & 2B butuh lirik)
    if fid not in lyrics_map: continue
    
    true_label = os.path.basename(os.path.dirname(path)).lower()
    if true_label not in TARGET_MOODS: continue
    
    try:
        # Load Audio Once
        y, sr = librosa.load(path, sr=16000)
        lyric = lyrics_map[fid]
        
        # --- STAGE 1: GATEKEEPER (HIGH vs LOW) ---
        y_trim = trim_middle(y, sr)
        feat1 = extract_s1(y_trim, sr).reshape(1, -1)
        p1 = model_stage1.predict(feat1, verbose=0)[0]
        
        # Mapping NN: Index 0=High, Index 1=Low
        is_high = p1[0] > p1[1] 
        
        pred_label = ""
        
        if is_high:
            # === JALUR A: HIGH ENERGY (ANGRY/HAPPY) ===
            feat2a = extract_s2a_audio(y, sr).reshape(1, -1)
            prob_aud = model_s2a_base.predict_proba(feat2a)[0]
            prob_txt = np.array(extract_s2a_text(lyric))
            
            # Meta Learner Stacking
            meta_in = np.concatenate([prob_aud, prob_txt]).reshape(1, -1)
            final_p = model_s2a_meta.predict_proba(meta_in)[0]
            
            pred_label = "angry" if final_p[0] > final_p[1] else "happy"
            
        else:
            # === JALUR B: LOW ENERGY (SAD/RELAXED) ===
            # Menggunakan Purified Random Forest (87% Acc)
            feat2b = extract_s2b(y, sr, lyric).reshape(1, -1)
            p2b = model_s2b.predict_proba(feat2b)[0]
            
            # Label RF Purified: 0=Sad, 1=Relaxed
            pred_label = "sad" if p2b[0] > p2b[1] else "relaxed"
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
    except Exception as e:
        print(f"Err {os.path.basename(path)}: {e}")

# ================= 4. REPORTING =================
print("\n" + "="*60)
print("📊 FINAL GLOBAL SYSTEM REPORT")
print("="*60)

# Global Accuracy
acc = accuracy_score(y_true, y_pred) * 100
print(f"🏆 GLOBAL ACCURACY: {acc:.2f}%")
print("-" * 60)

# Classification Report
print(classification_report(y_true, y_pred, target_names=TARGET_MOODS))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=TARGET_MOODS)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[x.upper() for x in TARGET_MOODS], 
            yticklabels=[x.upper() for x in TARGET_MOODS])
plt.xlabel('PREDIKSI MODEL')
plt.ylabel('LABEL ASLI')
plt.title(f'Global System Confusion Matrix\nAccuracy: {acc:.2f}%')
plt.tight_layout()
plt.savefig('global_final_cm.png')
plt.show()

print("✅ Evaluasi Selesai! Simpan 'global_final_cm.png'.")