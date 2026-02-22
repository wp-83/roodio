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
MODEL_PATHS = {
    # Stage 1: Neural Network (High vs Low)
    "stage1": "models/stage1_nn.h5",              
    
    # Stage 2A: Stacking Ensemble (Angry vs Happy)
    "stage2a_base": "models/stage2a_rf.pkl",   # Random Forest (Audio)
    "stage2a_meta": "models/stage2a_meta.pkl", # Meta Learner (Gabungan)
    
    # Stage 2B: SVM Pipeline (Sad vs Relaxed)
    "stage2b": "models/stage2b_tuned_final.pkl"     
}

print("🚀 MEMUAT SISTEM PREDIKSI HIERARKIS (STACKING VERSION)...")

# ================= 1. LOAD ALL MODELS =================
try:
    print("⏳ Loading Feature Extractors (YAMNet & RoBERTa)...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    nlp_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

    print("⏳ Loading Stage 1 Model (High/Low)...")
    model_stage1 = tf.keras.models.load_model(MODEL_PATHS["stage1"])

    print("⏳ Loading Stage 2A Models (Stacking)...")
    model_s2a_base = joblib.load(MODEL_PATHS["stage2a_base"]) # RF Audio
    model_s2a_meta = joblib.load(MODEL_PATHS["stage2a_meta"]) # Meta Learner

    print("⏳ Loading Stage 2B Model (Sad/Relaxed)...")
    model_stage2b = joblib.load(MODEL_PATHS["stage2b"])

    print("✅ SEMUA MODEL BERHASIL DIMUAT!\n")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")
    print(f"⚠️ Pastikan semua file model ada di folder 'models/'.")
    exit()

# ================= 2. FEATURE EXTRACTION FUNCTIONS =================

def extract_features_stage1(y, sr):
    """Fitur Stage 1: YAMNet Stats + RMS + ZCR"""
    try:
        if np.max(np.abs(y)) > 0: y_norm = y / np.max(np.abs(y))
        else: y_norm = y
        
        _, embeddings, _ = yamnet_model(y_norm)
        
        mean = tf.reduce_mean(embeddings, axis=0)
        std = tf.math.reduce_std(embeddings, axis=0)
        max_ = tf.reduce_max(embeddings, axis=0)
        yamnet_emb = tf.concat([mean, std, max_], axis=0).numpy()
        
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        return np.concatenate([yamnet_emb, [rms, zcr]])
    except: return None

def extract_features_stage2a_base(y, sr):
    """Fitur Stage 2A: YAMNet Mean + RMS + ZCR (Input untuk RF Base)"""
    _, embeddings, _ = yamnet_model(y) 
    yamnet_vec = tf.reduce_mean(embeddings, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([yamnet_vec, [rms, zcr]])

def extract_features_stage2b(y, sr):
    """Fitur Stage 2B: YAMNet Mean + Chroma + RMS (Input untuk SVM Pipeline)"""
    _, embeddings, _ = yamnet_model(y)
    yamnet_vec = tf.reduce_mean(embeddings, axis=0).numpy()
    chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    return np.concatenate([yamnet_vec, chroma, [rms]])

def get_text_prob_angry_happy(lyrics):
    """Mendapatkan probabilitas [Angry, Happy] dari Lirik"""
    if not lyrics or str(lyrics).strip() == "": 
        return [0.5, 0.5] 
    try:
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
    """Trim audio untuk Stage 1"""
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# ================= 3. PREDICTION ENGINE =================

def predict_mood(audio_path, lyrics_text=""):
    print(f"🎵 Analyzing: {os.path.basename(audio_path)}")
    
    # 1. LOAD AUDIO
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
    except Exception as e:
        print(f"❌ Error reading audio: {e}")
        return

    # --- LEVEL 1: HIGH vs LOW ENERGY ---
    y_trimmed = trim_middle(y, sr) 
    feat_s1 = extract_features_stage1(y_trimmed, sr).reshape(1, -1)
    
    # NN Output: [Prob_High, Prob_Low] (Karena mapping labelencoder: high=0, low=1)
    prob_s1 = model_stage1.predict(feat_s1, verbose=0)[0] 
    
    if prob_s1[0] > prob_s1[1]:
        energy_label = "High"
        confidence_s1 = prob_s1[0] * 100
    else:
        energy_label = "Low"
        confidence_s1 = prob_s1[1] * 100
    
    print(f"   🔹 Stage 1: {energy_label.upper()} ENERGY (Conf: {confidence_s1:.1f}%)")
    
    final_mood = ""
    final_conf = 0.0
    
    # --- LEVEL 2: SPECIFIC MOOD ---
    
    if energy_label == "High":
        print("   🔸 Masuk Branch High (Stacking): Angry vs Happy...")
        
        # --- STACKING INFERENCE LOGIC ---
        
        # 1. Base Prediction (Audio RF)
        feat_s2a = extract_features_stage2a_base(y, sr).reshape(1, -1)
        # Output: [Prob_Angry, Prob_Happy]
        prob_audio = model_s2a_base.predict_proba(feat_s2a)[0]
        
        # 2. Base Prediction (Text RoBERTa)
        prob_text = get_text_prob_angry_happy(lyrics_text)
        prob_text = np.array(prob_text) # Pastikan numpy array
        
        # 3. Create Meta Features (Input untuk Meta Learner)
        # Gabungkan [Prob_Audio_Angry, Prob_Audio_Happy, Prob_Text_Angry, Prob_Text_Happy]
        # Atau sesuai training: [Prob_Audio_Angry, Prob_Audio_Happy] + [Prob_Text_Angry, Prob_Text_Happy]
        meta_features = np.concatenate([prob_audio, prob_text]).reshape(1, -1)
        
        # 4. Meta Prediction (Hakim Ketua)
        # Output: [Prob_Angry, Prob_Happy]
        final_probs = model_s2a_meta.predict_proba(meta_features)[0]
        
        if final_probs[0] > final_probs[1]:
            final_mood = "Angry"
            final_conf = final_probs[0] * 100
        else:
            final_mood = "Happy"
            final_conf = final_probs[1] * 100
            
        print(f"      - Audio RF Opinion : Angry {prob_audio[0]:.2f} | Happy {prob_audio[1]:.2f}")
        print(f"      - Text Opinion     : Angry {prob_text[0]:.2f} | Happy {prob_text[1]:.2f}")
        print(f"      - Meta Decision    : {final_mood} ({final_conf:.1f}%)")
        
    else:
        print("   🔸 Masuk Branch Low (SVM Pipeline): Sad vs Relaxed...")
        
        # --- SVM PIPELINE INFERENCE ---
        feat_s2b = extract_features_stage2b(y, sr).reshape(1, -1)
        
        # Pipeline handle scaling & RFE automatically
        probs = model_stage2b.predict_proba(feat_s2b)[0]
        # Labels: 0=Sad, 1=Relaxed
        
        if probs[0] > probs[1]:
            final_mood = "Sad"
            final_conf = probs[0] * 100
        else:
            final_mood = "Relaxed"
            final_conf = probs[1] * 100

    # --- RESULT ---
    bar_len = int(final_conf / 5)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    
    print("\n" + "="*50)
    print(f"🎯 FINAL PREDICTION: {final_mood.upper()}")
    print(f"📊 Confidence      : [{bar}] {final_conf:.2f}%")
    print("="*50 + "\n")
    return final_mood

# ================= 4. RUNNER =================
if __name__ == "__main__":
    TEST_DIR = 'data/test_samples'
    
    if os.path.exists(TEST_DIR):
        files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.wav', '.mp3'))]
        if files:
            print(f"📂 Memproses {len(files)} file di '{TEST_DIR}'...\n")
            for f in files:
                path = os.path.join(TEST_DIR, f)
                
                # Cek apakah ada file lirik (.txt) dengan nama sama
                lyrics_path = path.replace('.wav', '.txt').replace('.mp3', '.txt')
                lyrics = ""
                if os.path.exists(lyrics_path):
                    with open(lyrics_path, 'r', encoding='utf-8') as lf:
                        lyrics = lf.read()
                
                predict_mood(path, lyrics_text=lyrics)
        else:
            print("⚠️ Folder test kosong.")
    else:
        os.makedirs(TEST_DIR)
        print(f"⚠️ Folder '{TEST_DIR}' dibuat. Silakan isi lagu.")