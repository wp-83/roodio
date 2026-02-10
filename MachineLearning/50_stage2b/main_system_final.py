import os
import re
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from transformers import pipeline

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🚀 MEMUAT SISTEM FINAL (HIERARCHICAL MULTIMODAL AI)...")
print("   Arsitektur: Stage 1 (NN) -> Stage 2A (Stacking) / Stage 2B (Purified RF)")

# ================= CONFIGURATION =================
MODEL_PATHS = {
    "stage1": "models/stage1_nn.h5",              
    "stage2a_base": "models/stage2a_rf.pkl",   
    "stage2a_meta": "models/stage2a_meta.pkl", 
    "stage2b": "models/stage2b_purified_rf.pkl" # SI JUARA 87%
}

# ================= 1. LOAD ALL MODELS =================
try:
    print("⏳ Loading Extractors (YAMNet & RoBERTa)...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

    print("⏳ Loading Stage Models...")
    model_stage1 = tf.keras.models.load_model(MODEL_PATHS["stage1"])
    model_s2a_base = joblib.load(MODEL_PATHS["stage2a_base"])
    model_s2a_meta = joblib.load(MODEL_PATHS["stage2a_meta"])
    model_s2b = joblib.load(MODEL_PATHS["stage2b"]) # Load model purified

    print("✅ SYSTEM READY TO SERVE!")

except Exception as e:
    print(f"\n❌ Error loading models: {e}")
    print("   Pastikan semua file .pkl dan .h5 ada di folder 'models/'.")
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

# ================= 3. MAIN PREDICTION ENGINE =================
def predict_song(audio_path, lyrics_text=""):
    filename = os.path.basename(audio_path)
    print(f"\n🎵 ANALYZING: {filename}")
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except:
        print("❌ Gagal membaca file audio")
        return

    # --- STAGE 1: HIGH vs LOW ---
    y_trim = trim_middle(y, sr)
    feat1 = extract_s1(y_trim, sr).reshape(1, -1)
    p1 = model_stage1.predict(feat1, verbose=0)[0]
    
    # Logic NN: Misal Index 0=High, Index 1=Low (Cek ulang jika terbalik)
    is_high = p1[0] > p1[1] 
    
    final_mood = ""
    confidence = 0.0
    stage_info = ""

    if is_high:
        # === JALUR A: HIGH ENERGY (ANGRY/HAPPY) ===
        feat2a = extract_s2a_audio(y, sr).reshape(1, -1)
        prob_aud = model_s2a_base.predict_proba(feat2a)[0]
        prob_txt = np.array(extract_s2a_text(lyrics_text))
        
        # Meta Learner
        meta_in = np.concatenate([prob_aud, prob_txt]).reshape(1, -1)
        final_p = model_s2a_meta.predict_proba(meta_in)[0]
        
        if final_p[0] > final_p[1]:
            final_mood = "Angry"
            confidence = final_p[0]
        else:
            final_mood = "Happy"
            confidence = final_p[1]
            
        stage_info = f"Stage 2A (Stacking) | AudConf:{prob_aud.max():.2f}"

    else:
        # === JALUR B: LOW ENERGY (SAD/RELAXED) ===
        # Menggunakan Model Purified RF (87% Accuracy)
        feat2b = extract_s2b(y, sr, lyrics_text).reshape(1, -1)
        final_p = model_s2b.predict_proba(feat2b)[0]
        
        if final_p[0] > final_p[1]:
            final_mood = "Sad"
            confidence = final_p[0]
        else:
            final_mood = "Relaxed"
            confidence = final_p[1]
            
        stage_info = f"Stage 2B (Purified RF) | Acc:87%"

    # --- FINAL OUTPUT ---
    bar = "█" * int(confidence * 20)
    print("\n" + "="*50)
    print(f"🎯 HASIL PREDIKSI: {final_mood.upper()}")
    print(f"📊 Confidence    : {confidence*100:.1f}%  |{bar:<20}|")
    print("-" * 50)
    print(f"🛠️  Engine Info   : {stage_info}")
    print("="*50 + "\n")
    return final_mood

# ================= 4. CONTOH PENGGUNAAN =================
if __name__ == "__main__":
    # Ganti path ini untuk tes
    TEST_FILE = "data/raw/sad/151_Adele - Hello (Official Music Video).wav"
    TEST_LYRIC = "Hello, it's me. I was wondering if after all these years you'd like to meet."
    
    if os.path.exists(TEST_FILE):
        predict_song(TEST_FILE, TEST_LYRIC)
    else:
        print("⚠️ File tes tidak ditemukan. Ganti path di bagian bawah script.")