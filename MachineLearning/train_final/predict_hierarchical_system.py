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

# ================= CONFIGURATION =================
# Pastikan path model ini benar sesuai dengan yang sudah kamu save
MODEL_PATHS = {
    "stage1": "models/stage1_nn.h5",              # Neural Network (High vs Low)
    "stage2a_audio": "models/stage2a_audio_rf.pkl", # Random Forest (Angry vs Happy Audio)
    # Stage 2A Text tidak perlu diload manual, pakai HuggingFace pipeline langsung
    "stage2b": "models/stage2b_tuned_final.pkl"     # SVM Pipeline (Sad vs Relaxed)
}

# Bobot Ensemble Stage 2A (Angry vs Happy)
# Ganti nilai ini sesuai hasil 'train_stage2a_weighted_ensemble.py' kamu
STAGE2A_WEIGHT_AUDIO = 0.6  # Contoh: Audio 60%
STAGE2A_WEIGHT_TEXT = 0.4   # Contoh: Text 40%

print("🚀 MEMUAT SISTEM PREDIKSI HIERARKIS...")

# ================= 1. LOAD MODELS =================
try:
    # Load YAMNet (Shared Feature Extractor)
    print("⏳ Loading YAMNet...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    # Load RoBERTa (Text Feature Extractor)
    print("⏳ Loading RoBERTa...")
    nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

    # Load Stage 1 Model (Keras)
    print("⏳ Loading Stage 1 (High/Low)...")
    model_stage1 = tf.keras.models.load_model(MODEL_PATHS["stage1"])

    # Load Stage 2A Model (Sklearn)
    print("⏳ Loading Stage 2A (Angry/Happy)...")
    model_stage2a_audio = joblib.load(MODEL_PATHS["stage2a_audio"])

    # Load Stage 2B Model (Sklearn Pipeline)
    print("⏳ Loading Stage 2B (Sad/Relaxed)...")
    model_stage2b = joblib.load(MODEL_PATHS["stage2b"])

    print("✅ SEMUA MODEL BERHASIL DIMUAT!\n")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("   Pastikan semua file model ada di folder 'models/'")
    exit()

# ================= 2. FEATURE EXTRACTION FUNCTIONS =================

def extract_features_stage1(y, sr):
    """Fitur untuk Stage 1 (YAMNet Mean/Std/Max + RMS + ZCR)"""
    try:
        # Preprocessing khusus Stage 1 (Normalize)
        if np.max(np.abs(y)) > 0: y_norm = y / np.max(np.abs(y))
        else: y_norm = y
        
        # YAMNet Embedding
        _, embeddings, _ = yamnet_model(y_norm)
        
        # Statistik Embedding
        mean = tf.reduce_mean(embeddings, axis=0)
        std = tf.math.reduce_std(embeddings, axis=0)
        max_ = tf.reduce_max(embeddings, axis=0)
        yamnet_emb = tf.concat([mean, std, max_], axis=0).numpy()
        
        # Fitur Klasik
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        return np.concatenate([yamnet_emb, [rms, zcr]])
    except: return None

def extract_features_stage2a_audio(y, sr):
    """Fitur untuk Stage 2A (YAMNet Mean + RMS + ZCR)"""
    # Note: Stage 2A pakai Random Forest yang dilatih dengan fitur ini
    _, embeddings, _ = yamnet_model(y) # Tanpa normalize aneh-aneh
    yamnet_vec = tf.reduce_mean(embeddings, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([yamnet_vec, [rms, zcr]])

def extract_features_stage2b(y, sr):
    """Fitur untuk Stage 2B (YAMNet Mean + Chroma + RMS)"""
    _, embeddings, _ = yamnet_model(y)
    yamnet_vec = tf.reduce_mean(embeddings, axis=0).numpy()
    chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    return np.concatenate([yamnet_vec, chroma, [rms]])

def get_text_prob_angry_happy(lyrics):
    """Mendapatkan probabilitas [Angry, Happy] dari Lirik"""
    if not lyrics: return [0.5, 0.5] # Netral jika tidak ada lirik
    
    try:
        # Cleaning Text (Sederhana)
        text = str(lyrics).lower()
        text = re.sub(r"[^a-z0-9\s]", '', text)
        
        output = nlp_classifier(text[:512])[0]
        scores = {item['label']: item['score'] for item in output}
        
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
        
        total = s_happy + s_angry + 1e-9
        return [s_angry/total, s_happy/total]
    except: return [0.5, 0.5]

def trim_middle(y, sr, percentage=0.5):
    """Helper untuk Stage 1 (mirip saat training)"""
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# ================= 3. MAIN PREDICTION LOGIC =================

def predict_mood(audio_path, lyrics_text=""):
    print(f"🎵 Analyzing: {os.path.basename(audio_path)}")
    
    # A. Load Audio (Sekali untuk semua)
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
    except Exception as e:
        print(f"❌ Error reading audio: {e}")
        return "Error"

    # --- LEVEL 1: HIGH vs LOW ENERGY ---
    # Siapkan fitur Stage 1 (Trimmed Audio)
    y_trimmed = trim_middle(y, sr) 
    feat_s1 = extract_features_stage1(y_trimmed, sr)
    
    # Reshape (1, n_features)
    feat_s1 = feat_s1.reshape(1, -1)
    
    # Prediksi Stage 1
    prob_s1 = model_stage1.predict(feat_s1, verbose=0)[0] # [Prob_Low, Prob_High]
    energy_label = "High" if prob_s1[1] > prob_s1[0] else "Low"
    confidence_s1 = max(prob_s1) * 100
    
    print(f"   🔹 Stage 1: {energy_label.upper()} ENERGY ({confidence_s1:.1f}%)")
    
    final_mood = ""
    final_conf = 0.0
    
    # --- LEVEL 2: SPECIFIC MOOD ---
    
    if energy_label == "High":
        # === BRANCH A: ANGRY vs HAPPY ===
        print("   🔸 Masuk Branch High: Cek Angry vs Happy...")
        
        # 1. Audio Prediction
        feat_s2a = extract_features_stage2a_audio(y, sr).reshape(1, -1)
        prob_audio = model_stage2a_audio.predict_proba(feat_s2a)[0] # [Prob_Angry, Prob_Happy]
        
        # 2. Text Prediction
        prob_text = get_text_prob_angry_happy(lyrics_text) # [Prob_Angry, Prob_Happy]
        
        # 3. Weighted Ensemble
        # Rumus: Final = (W_aud * P_aud) + (W_txt * P_txt)
        final_prob_angry = (STAGE2A_WEIGHT_AUDIO * prob_audio[0]) + (STAGE2A_WEIGHT_TEXT * prob_text[0])
        final_prob_happy = (STAGE2A_WEIGHT_AUDIO * prob_audio[1]) + (STAGE2A_WEIGHT_TEXT * prob_text[1])
        
        if final_prob_angry > final_prob_happy:
            final_mood = "Angry"
            final_conf = final_prob_angry * 100
        else:
            final_mood = "Happy"
            final_conf = final_prob_happy * 100
            
        print(f"      - Audio Prob : Angry {prob_audio[0]:.2f} | Happy {prob_audio[1]:.2f}")
        print(f"      - Text Prob  : Angry {prob_text[0]:.2f} | Happy {prob_text[1]:.2f}")
        
    else:
        # === BRANCH B: SAD vs RELAXED ===
        print("   🔸 Masuk Branch Low: Cek Sad vs Relaxed...")
        
        # 1. Feature Extraction (Pipeline Stage 2B akan handle scaling & RFE)
        feat_s2b = extract_features_stage2b(y, sr).reshape(1, -1)
        
        # 2. Prediction (Pipeline)
        # Ingat label training kita: 0=Sad, 1=Relaxed (perlu dicek urutan classes_)
        classes = model_stage2b.classes_ # Biasanya [0, 1]
        probs = model_stage2b.predict_proba(feat_s2b)[0]
        
        # Mapping: Asumsi 0=Sad, 1=Relaxed (berdasarkan alfabet/training script sebelumnya)
        # Jika training kamu pakai LabelEncoder otomatis, urutan alfabet: Relaxed(0), Sad(1)
        # TAPI di script manual kita define: 0=Sad, 1=Relaxed. Mari kita pakai asumsi training script terakhir.
        # Di script terakhir: labels = [0]*len(sad) + [1]*len(relaxed) -> 0=Sad, 1=Relaxed.
        
        prob_sad = probs[0]
        prob_relaxed = probs[1]
        
        if prob_sad > prob_relaxed:
            final_mood = "Sad"
            final_conf = prob_sad * 100
        else:
            final_mood = "Relaxed"
            final_conf = prob_relaxed * 100
            
    # --- FINAL OUTPUT ---
    print("\n" + "="*40)
    print(f"🎯 FINAL PREDICTION: {final_mood.upper()}")
    print(f"📊 Confidence      : {final_conf:.2f}%")
    print("="*40 + "\n")
    return final_mood

# ================= 4. TEST EXECUTION =================
if __name__ == "__main__":
    # Contoh Penggunaan
    # Buat folder 'data/test_samples' dan isi lagu untuk mengetes
    test_dir = 'data/test_samples'
    
    if os.path.exists(test_dir):
        files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3'))]
        for f in files:
            path = os.path.join(test_dir, f)
            # Contoh lirik dummy (ideally load from txt file with same name)
            dummy_lyrics = "I am so happy today the sun is shining" 
            predict_mood(path, lyrics_text=dummy_lyrics)
    else:
        print("⚠️ Folder test tidak ditemukan. Silakan input path manual.")
        # while True:
        #     p = input("Path Audio: ")
        #     if p == 'q': break
        #     l = input("Lirik (Optional): ")
        #     predict_mood(p, l)