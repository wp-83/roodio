import os
import numpy as np
import librosa
import joblib 
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline
import logging

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

print("🚀 INITIALIZING FINAL INFERENCE SYSTEM (TUNED)...")
print("="*50)

# --- 1. CONFIG & RESOURCES ---
MODEL_DIR = 'models/'
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
ROBERTA_MODEL = "j-hartmann/emotion-english-distilroberta-base"

print("⏳ Loading AI Models...")
yamnet_model = hub.load(YAMNET_URL)
nlp_classifier = pipeline("text-classification", model=ROBERTA_MODEL, top_k=None, truncation=True)

# --- 2. LOAD TRAINED CLASSIFIERS ---
try:
    # STAGE 1: High vs Low (NN)
    s1_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'stage1_nn.h5'))
    
    # STAGE 2A: Angry vs Happy (RF + LR)
    s2a_rf = joblib.load(os.path.join(MODEL_DIR, 'stage2a_rf.pkl'))
    s2a_meta = joblib.load(os.path.join(MODEL_DIR, 'stage2a_meta.pkl'))
    
    # STAGE 2B: Sad vs Relaxed (RF + SVM + Scaler) - UPDATE INI PENTING!
    s2b_rf = joblib.load(os.path.join(MODEL_DIR, 'stage2b_rf.pkl'))
    s2b_meta = joblib.load(os.path.join(MODEL_DIR, 'stage2b_meta.pkl'))
    s2b_scaler = joblib.load(os.path.join(MODEL_DIR, 'stage2b_scaler.pkl'))
    
    print("✅ Semua model & scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Pastikan folder 'models/' lengkap (termasuk stage2b_scaler.pkl)")
    exit()

# --- 3. FEATURE EXTRACTION FUNCTIONS ---

def trim_middle(y, sr=16000, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# FITUR STAGE 1 (Sesuai Training Exp 17)
def extract_feat_stage1(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        y = trim_middle(y, sr)
        if np.max(np.abs(y)) > 0: y_norm = y / np.max(np.abs(y))
        else: y_norm = y
        if len(y_norm) < 16000: y_norm = np.pad(y_norm, (0, 16000 - len(y_norm)))
            
        _, embeddings, _ = yamnet_model(y_norm)
        mean = tf.reduce_mean(embeddings, axis=0)
        std = tf.math.reduce_std(embeddings, axis=0)
        max_ = tf.reduce_max(embeddings, axis=0)
        yamnet_emb = tf.concat([mean, std, max_], axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_emb, [rms, zcr]]).reshape(1, -1)
    except: return None

# FITUR STAGE 2A: ANGRY/HAPPY (Simple Mean)
def extract_feat_s2a(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, [rms, zcr]]).reshape(1, -1)
    except: return None

# FITUR STAGE 2B: SAD/RELAXED (Mean + Std + RMS + ZCR) - UPDATE!
def extract_feat_s2b(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        # Tambahan STD Deviasi (Sesuai Training Exp 29)
        yamnet_std = tf.math.reduce_std(emb, axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, yamnet_std, [rms, zcr]]).reshape(1, -1)
    except: return None

# FITUR TEKS: 2A (Simple 2 Scores) vs 2B (Full 7 Scores)
def extract_text_feat(lyrics, mode):
    try:
        chunk = str(lyrics)[:512]
        output = nlp_classifier(chunk)[0]
        s = {item['label']: item['score'] for item in output}
        
        if mode == '2a': # Angry vs Happy (Simple)
            val0 = s.get('anger',0) + s.get('disgust',0) + s.get('fear',0) + s.get('sadness', 0)
            val1 = s.get('joy',0) + s.get('surprise',0)
            return np.array([[val0, val1]])
            
        elif mode == '2b': # Sad vs Relaxed (Full 7 Emotions for SVM)
            # Urutan abjad: anger, disgust, fear, joy, neutral, sadness, surprise
            sorted_keys = sorted(s.keys())
            scores = [s[k] for k in sorted_keys]
            # Pastikan jika ada key hilang, diisi 0 (biasanya Roberta output lengkap, tapi jaga2)
            if len(scores) < 7: return np.array([[0.0]*7])
            return np.array([scores])
            
    except: 
        if mode=='2a': return np.array([[0.5, 0.5]])
        else: return np.array([[0.0]*7])

# --- 4. CLASSIFICATION LOGIC ---

def classify_song(audio_path, lyrics):
    print(f"\n🎵 Menganalisis: {os.path.basename(audio_path)}")
    print("-" * 40)
    
    # === STAGE 1: HIGH vs LOW ===
    feat_s1 = extract_feat_stage1(audio_path)
    if feat_s1 is None: return "Error Audio"
    
    prob_s1 = s1_model.predict(feat_s1, verbose=0)[0]
    stage1_pred = np.argmax(prob_s1) # 0=High, 1=Low (Check label encoder training!)
    
    if stage1_pred == 0: # HIGH AROUSAL BRANCH
        print(f"📍 Stage 1: HIGH ENERGY ({prob_s1[0]*100:.1f}%) -> Routing to Angry/Happy")
        
        # Ekstraksi Fitur Stage 2A
        feat_aud = extract_feat_s2a(audio_path)
        feat_txt = extract_text_feat(lyrics, '2a')
        
        # Prediksi RF
        prob_rf = s2a_rf.predict_proba(feat_aud)
        # Prediksi Meta (LR)
        meta_input = np.concatenate([prob_rf, feat_txt], axis=1)
        final_idx = s2a_meta.predict(meta_input)[0]
        final_prob = s2a_meta.predict_proba(meta_input)[0][final_idx]
        
        result = "ANGRY 😡" if final_idx == 0 else "HAPPY 😄"
        
    else: # LOW AROUSAL BRANCH
        print(f"📍 Stage 1: LOW ENERGY ({prob_s1[1]*100:.1f}%) -> Routing to Sad/Relaxed")
        
        # Ekstraksi Fitur Stage 2B (Complex)
        feat_aud = extract_feat_s2b(audio_path) # Ada Std Dev
        feat_txt = extract_text_feat(lyrics, '2b') # 7 Emosi
        
        # Prediksi RF (Probabilitas dari cross_val di training)
        # Note: RF kita predict_proba
        prob_rf = s2b_rf.predict_proba(feat_aud) 
        
        # Gabung Input Meta
        meta_input_raw = np.concatenate([prob_rf, feat_txt], axis=1)
        
        # SCALING (Wajib untuk SVM!)
        meta_input_scaled = s2b_scaler.transform(meta_input_raw)
        
        # Prediksi Meta (SVM)
        final_idx = s2b_meta.predict(meta_input_scaled)[0]
        # SVM probability=True harus aktif saat training untuk ini
        try:
            final_prob = s2b_meta.predict_proba(meta_input_scaled)[0][final_idx]
        except:
            final_prob = 1.0 # Fallback jika probability=False
            
        result = "SAD 😢" if final_idx == 0 else "RELAXED 😌"

    print("-" * 40)
    print(f"🏆 HASIL AKHIR: {result}")
    print(f"🎯 Keyakinan  : {final_prob*100:.2f}%")
    return result

# --- 5. EXECUTION ---
# Ganti path file audio & lirik di sini
TEST_AUDIO = r"C:\Users\andiz\Downloads\Meghan Trainor - Made You Look (Official Music Video).mp3"
TEST_LYRICS = """
I could have my Gucci on
I could wear my Louis Vuitton
But even with nothing on
Bet I made you look (I made you look)
I'll make you double take soon as I walk away
Call up your chiropractor just in case your neck break
Ooh, tell me what you, what you, what you gon' do? Ooh
'Cause I'm 'bout to make a scene, double up that sunscreen
I'm 'bout to turn the heat up, gonna make your glasses steam
Ooh, tell me what you, what you, what you gon' do? Ooh
When I do my walk, walk
I can guarantee your jaw will drop, drop
'Cause they don't make a lot of what I got, got
Ladies, if you feel me, this your bop, bop
(Bop, bop, bop)
I could have my Gucci on (Gucci on)
I could wear my Louis Vuitton
But even with nothing on
Bet I made you look (I made you look)
Yeah, I look good in my Versace dress (take it off)
But I'm hotter when my morning hair's a mess
But even with my hoodie on
Bet I made you look (I made you look)
(Hmm-hmm-hmm)
And once you get a taste, you'll never be the same
This ain't that ordinary, it's that 14 karat cake
Ooh, tell me what you, what you
What you gon' do? Ooh
When I do my walk, walk
I can guarantee your jaw will drop, drop
'Cause they don't make a lot of what I got, got
Ladies, if you feel me, this your bop, bop
(Bop, bop, bop)
I could have my Gucci on (Gucci on)
I could wear my Louis Vuitton
But even with nothing on
Bet I made you look (said I made you look)
Yeah, I look good in my Versace dress (take it off)
But I'm hotter when my morning hair's a mess
But even with my hoodie on
Bet I made you look (said, I made you look)
"""

if os.path.exists(TEST_AUDIO):
    classify_song(TEST_AUDIO, TEST_LYRICS)
else:
    print(f"⚠️ File audio tidak ditemukan di: {TEST_AUDIO}")
    # Opsi: Buat dummy file untuk test logic code
    # import soundfile as sf; sf.write("dummy_test.wav", np.random.uniform(-0.1,0.1,16000*5), 16000); classify_song("dummy_test.wav", TEST_LYRICS)