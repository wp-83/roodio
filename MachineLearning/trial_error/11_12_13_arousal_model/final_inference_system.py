import os
import numpy as np
import librosa
import joblib  # Untuk load model sklearn
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline
import logging

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

print("🚀 INITIALIZING HIERARCHICAL INFERENCE SYSTEM...")
print("="*50)

# --- 1. CONFIG & RESOURCES ---
MODEL_DIR = 'models/'
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
ROBERTA_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Load Feature Extractors (Hanya perlu sekali load)
print("⏳ Loading YAMNet & RoBERTa...")
yamnet_model = hub.load(YAMNET_URL)
nlp_classifier = pipeline("text-classification", model=ROBERTA_MODEL, top_k=None, truncation=True)

# --- 2. LOAD TRAINED MODELS ---
print("⏳ Loading Trained Classifiers...")

try:
    # STAGE 1: High vs Low (Keras Neural Network)
    s1_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'stage1_nn.h5'))
    
    # STAGE 2A: Angry vs Happy (Stacking RF + LR)
    s2a_rf = joblib.load(os.path.join(MODEL_DIR, 'stage2a_rf.pkl'))
    s2a_meta = joblib.load(os.path.join(MODEL_DIR, 'stage2a_meta.pkl'))
    
    # STAGE 2B: Sad vs Relaxed (Stacking RF + LR)
    s2b_rf = joblib.load(os.path.join(MODEL_DIR, 'stage2b_rf.pkl'))
    s2b_meta = joblib.load(os.path.join(MODEL_DIR, 'stage2b_meta.pkl'))
    
    print("✅ Semua model berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Pastikan folder 'models/' berisi: stage1_nn.h5, stage2a_rf.pkl, stage2a_meta.pkl, stage2b_rf.pkl, stage2b_meta.pkl")
    exit()

# --- 3. FEATURE EXTRACTION FUNCTIONS ---

# Helper: Trim Middle (Persis seperti di Exp 17)
def trim_middle(y, sr=16000, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# FITUR KHUSUS STAGE 1 (Complex: Mean + Std + Max)
def extract_feat_stage1(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        y = trim_middle(y, sr) 
        
        # Normalisasi
        if np.max(np.abs(y)) > 0: y_norm = y / np.max(np.abs(y))
        else: y_norm = y
        if len(y_norm) < 16000: y_norm = np.pad(y_norm, (0, 16000 - len(y_norm)))
            
        _, embeddings, _ = yamnet_model(y_norm)
        
        # Statistik YAMNet (Sesuai Exp 17)
        mean = tf.reduce_mean(embeddings, axis=0)
        std = tf.math.reduce_std(embeddings, axis=0)
        max_ = tf.reduce_max(embeddings, axis=0)
        yamnet_emb = tf.concat([mean, std, max_], axis=0).numpy()
        
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # Output shape (1, 3074)
        return np.concatenate([yamnet_emb, [rms, zcr]]).reshape(1, -1)
    except: return None

# FITUR KHUSUS STAGE 2 (Simple: Mean Only)
def extract_feat_stage2_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        
        _, emb, _ = yamnet_model(y)
        # Sesuai Exp 27 (Hanya Mean)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # Output shape (1, 1026)
        return np.concatenate([yamnet_vec, [rms, zcr]]).reshape(1, -1)
    except: return None

# FITUR TEKS (ROBERTA SCORE)
def extract_feat_stage2_text(lyrics, mode):
    try:
        chunk = str(lyrics)[:512]
        output = nlp_classifier(chunk)[0]
        s = {item['label']: item['score'] for item in output}
        
        if mode == 'angry_happy':
            # Exp 27 Logic: Angry vs Happy
            val0 = s.get('anger',0) + s.get('disgust',0) + s.get('fear',0) + s.get('sadness', 0) # Angry Group
            val1 = s.get('joy',0) + s.get('surprise',0) # Happy Group
        elif mode == 'sad_relaxed':
            # Exp 27 Logic: Sad vs Relaxed
            val0 = s.get('sadness',0) + s.get('fear',0) + s.get('anger',0) + s.get('disgust',0) # Sad Group
            val1 = s.get('neutral',0) + s.get('joy',0) + s.get('surprise',0) # Relaxed Group
            
        return np.array([[val0, val1]])
    except: return np.array([[0.5, 0.5]])

# --- 4. PREDICTION LOGIC ---

def predict_stacking(audio_feat, text_score, rf_model, meta_model):
    # 1. Base Model Audio (RF)
    prob_audio = rf_model.predict_proba(audio_feat) # [[prob_class0, prob_class1]]
    
    # 2. Gabung Input Meta [Prob_Audio, Score_Text]
    meta_input = np.concatenate([prob_audio, text_score], axis=1)
    
    # 3. Meta Prediction (LR)
    final_prob = meta_model.predict_proba(meta_input)[0]
    final_pred = np.argmax(final_prob)
    
    return final_pred, final_prob

def classify_song(audio_path, lyrics):
    print(f"\n🎵 Menganalisis Lagu: {os.path.basename(audio_path)}")
    print("-" * 40)
    
    # === STAGE 1: HIGH vs LOW (Neural Network) ===
    # Labelencoder biasanya: 'high' (0), 'low' (1)
    # Kita asumsikan urutan abjad default sklearn LabelEncoder
    
    feat_s1 = extract_feat_stage1(audio_path)
    if feat_s1 is None: return "Error Audio"
    
    # Prediksi NN
    prob_s1 = s1_model.predict(feat_s1, verbose=0)[0] 
    stage1_pred = np.argmax(prob_s1)
    
    # Logika Label Encoder: 'high' datang sebelum 'low' -> 0=High, 1=Low
    # Jika di training Exp 17 kamu urutannya beda, tukar logika if ini.
    
    if stage1_pred == 0: # HIGH AROUSAL
        conf = prob_s1[0]
        print(f"📍 Stage 1: HIGH ENERGY Detected ({conf*100:.1f}%)")
        print("   ↳ Masuk ke Model Angry/Happy...")
        
        # === STAGE 2A: ANGRY vs HAPPY ===
        # 0 = Angry, 1 = Happy
        feat_aud = extract_feat_stage2_audio(audio_path)
        feat_txt = extract_feat_stage2_text(lyrics, 'angry_happy')
        
        pred, probs = predict_stacking(feat_aud, feat_txt, s2a_rf, s2a_meta)
        result = "ANGRY 😡" if pred == 0 else "HAPPY 😄"
        final_conf = probs[pred]
        
    else: # LOW AROUSAL
        conf = prob_s1[1]
        print(f"📍 Stage 1: LOW ENERGY Detected ({conf*100:.1f}%)")
        print("   ↳ Masuk ke Model Sad/Relaxed...")
        
        # === STAGE 2B: SAD vs RELAXED ===
        # 0 = Sad, 1 = Relaxed
        feat_aud = extract_feat_stage2_audio(audio_path)
        feat_txt = extract_feat_stage2_text(lyrics, 'sad_relaxed')
        
        pred, probs = predict_stacking(feat_aud, feat_txt, s2b_rf, s2b_meta)
        result = "SAD 😢" if pred == 0 else "RELAXED 😌"
        final_conf = probs[pred]
        
    print("-" * 40)
    print(f"🏆 HASIL AKHIR: {result}")
    print(f"🎯 Keyakinan  : {final_conf*100:.2f}%")
    return result

# --- 5. TESTING AREA ---
# Ganti ini dengan file audio & lirik tes kamu
sample_audio = r"C:\Users\andiz\Downloads\In The End [Official HD Music Video] - Linkin Park.mp3" 
sample_lyrics = "It starts with one One thing, I don't know why It doesn't even matter how hard you try Keep that in mind, I designed this rhyme to explain in due time All I know time is a valuable thing Watch it fly by as the pendulum swings Watch it count down to the end of the day, the clock ticks life away It's so unreal, didn't look out below Watch the time go right out the window Tryna hold on, d-didn't even know I wasted it all just to watch you go I kept everything inside And even though I tried, it all fell apart What it meant to me will eventually be a memory of a time when I tried so hard and got so far But in the end, it doesn't even matter I had to fall to lose it all But in the end, it doesn't even matter One thing, I don't know why It doesn't even matter how hard you try Keep that in mind, I designed this rhyme to remind myself how I tried so hard In spite of the way you were mockin' me, actin' like I was part of your property Rememberin' all the times you fought with me I'm surprised it got so far Things aren't the way they were before You wouldn't even recognize me anymore Not that you knew me back then, but it all comes back to me in the end You kept everything inside And even though I tried, it all fell apart What it meant to me will eventually be a memory of a time when I tried so hard and got so far But in the end, it doesn't even matter I had to fall to lose it all But in the end, it doesn't even matter I've put my trust in you Pushed as far as I can go For all this, there's only one thing you should know I've put my trust in you Pushed as far as I can go For all this, there's only one thing you should know I tried so hard and got so far But in the end, it doesn't even matter I had to fall to lose it all But in the end, it doesn't even matter"

# Buat dummy file audio jika belum ada (Agar code bisa ditest run)
if not os.path.exists(sample_audio):
    print("⚠️ Membuat dummy audio file untuk testing...")
    import soundfile as sf
    dummy = np.random.uniform(-0.1, 0.1, 16000*5) # 5 detik noise
    sf.write(sample_audio, dummy, 16000)

# Jalankan Klasifikasi
if os.path.exists(sample_audio):
    classify_song(sample_audio, sample_lyrics)