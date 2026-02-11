import os
import re
import numpy as np
import librosa
import joblib 
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn # Tambahan untuk definisi class
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

print("🚀 MANUAL SAMPLE TESTING (Global System - PyTorch Edition)")
print("   Audio: Dari folder 'test_samples'")
print("   Lirik: Input Manual di Script")
print("="*60)

# ================= CONFIGURASI =================

# 1. Lokasi Folder Audio Sample
AUDIO_FOLDER = 'data/test_samples' 

# 2. INPUT LIRIK MANUAL DISINI (Format: "nama_file.mp3": "lirik lagu...")
MANUAL_DATA = {
    "Julien Baker - Appointments (Official Video).mp3": """
        I'm staying in tonight
        I won't stop you from leaving
        I know that I'm not what you wanted, am I?
        Maybe it's all gonna turn out alright
        And I know that it's not, but I have to believe that it is
    """,
    
    "Meghan Trainor - Made You Look (Official Music Video).mp3": """
        I could have my Gucci on
        I could wear my Louis Vuitton
        But even with nothing on
        Bet I made you look (I made you look)
    """,
    
    "Kesha - Praying (Official Video).mp3": """
        Well, you almost had me fooled
        Told me that I was nothing without you
        I hope you're somewhere praying, praying
        I hope your soul is changing, changing
    """,
    
    "In The End [Official HD Music Video] - Linkin Park.mp3": """
        It starts with one
        One thing I don't know why
        It doesn't even matter how hard you try
        I tried so hard and got so far
        But in the end it doesn't even matter
    """,
    
    "lagu_galau_tapi_santai.mp3": """
        And I hate to say I love you
        When it's so hard for me
        And I hate to say I want you
        When you make it so clear
        You don't want me
    """ 
}

# 3. Path Model
MODEL_DIR = 'models/v2' # Sesuaikan dengan lokasi .pth
BERT_PATH = 'models/model_hybrid_final' # Path Model BERT Stage 2B
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 1. DEFINISI ARSITEKTUR STAGE 1 (PYTORCH) =================
# Wajib ada agar PyTorch bisa me-load bobot model
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=3074): 
        super(AudioClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(256, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)

# ================= 2. LOAD MODELS =================
print("⏳ Loading AI Brains...")

try:
    # A. Audio Feature Extractor (YAMNet)
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # B. Stage 1: Audio Neural Network (PyTorch)
    s1_path = os.path.join(MODEL_DIR, 'stage1_nn.pth')
    s1_model = AudioClassifier().to(DEVICE)
    s1_model.load_state_dict(torch.load(s1_path, map_location=DEVICE))
    s1_model.eval()
    
    # Load Encoder Stage 1 (Untuk tahu 0=High atau Low)
    # Jika file tidak ada, kita asumsi default training saya: 0=High, 1=Low
    try:
        s1_encoder = joblib.load(os.path.join(MODEL_DIR, 'stage1_encoder.pkl'))
    except:
        s1_encoder = None
        print("⚠️ Warning: Encoder Stage 1 tidak ditemukan. Menggunakan default mapping.")

    # C. Stage 2A: Random Forest
    # Lokasi file ini mungkin beda, sesuaikan
    s2a_rf = joblib.load(os.path.join('models', 'stage2a_rf.pkl'))
    s2a_meta = joblib.load(os.path.join('models', 'stage2a_meta.pkl'))
    
    # D. Text Model (BERT Stage 2B)
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH).to(DEVICE)
    bert_model.eval()
    
    # ✅ FIX LABEL MAPPING (Sesuai temuan kamu: 0=sad, 1=relaxed)
    id2label = {0: 'sad', 1: 'relaxed'}
    
    print("✅ System Ready.")

except Exception as e:
    print(f"❌ Error Loading Models: {e}")
    exit()

# ================= 3. HELPER FUNCTIONS =================

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', str(text))
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def trim_middle(y, sr=16000, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def extract_feat_s1(y, sr):
    y_trim = trim_middle(y, sr)
    if np.max(np.abs(y_trim)) > 0: y_norm = y_trim / np.max(np.abs(y_trim))
    else: y_norm = y_trim
    if len(y_norm) < 16000: y_norm = np.pad(y_norm, (0, 16000 - len(y_norm)))
    _, embeddings, _ = yamnet_model(y_norm)
    
    # Konversi Tensor TF ke Numpy
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_std = tf.math.reduce_std(embeddings, axis=0).numpy()
    emb_max = tf.reduce_max(embeddings, axis=0).numpy()
    
    rms = np.mean(librosa.feature.rms(y=y_trim))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_trim))
    
    return np.concatenate([emb_mean, emb_std, emb_max, [rms, zcr]])

def extract_feat_s2a_audio(y, sr):
    if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
    _, emb, _ = yamnet_model(y)
    vec = tf.reduce_mean(emb, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([vec, [rms, zcr]]).reshape(1, -1)

# ================= 4. TESTING LOGIC =================

def predict_song(filename, lyrics_raw):
    path = os.path.join(AUDIO_FOLDER, filename)
    
    if not os.path.exists(path):
        print(f"❌ File Audio Tidak Ditemukan: {path}")
        return

    print("\n" + "-"*50)
    print(f"🎵 Playing: {filename}")
    print(f"📝 Lyrics : {lyrics_raw[:50].strip()}...") 

    try:
        # 1. Load Audio
        y, sr = librosa.load(path, sr=16000)
        
        # 2. Stage 1: Energy Check (PyTorch Inference)
        feat_s1 = extract_feat_s1(y, sr)
        feat_tensor = torch.tensor([feat_s1], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            logits = s1_model(feat_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            
        # Decode Label (High/Low)
        if s1_encoder:
            stage1_label = s1_encoder.inverse_transform([pred_idx])[0].lower()
        else:
            # Fallback manual jika encoder hilang (Asumsi 0=High, 1=Low)
            stage1_label = "high" if pred_idx == 0 else "low"
            
        # Cek High Energy
        is_high = 'high' in stage1_label or 'angry' in stage1_label or 'happy' in stage1_label
        
        if is_high:
            # ---> HIGH ENERGY (Angry/Happy)
            branch = "High Energy (Audio Focus)"
            f_aud = extract_feat_s2a_audio(y, sr)
            f_txt = np.array([[0.5, 0.5]]) # Dummy text
            
            p_rf = s2a_rf.predict_proba(f_aud)
            meta_in = np.concatenate([p_rf, f_txt], axis=1)
            idx = s2a_meta.predict(meta_in)[0]
            conf = s2a_meta.predict_proba(meta_in)[0][idx]
            
            # Mapping Manual RF (Sesuaikan jika hasil terbalik)
            mood = "ANGRY 😡" if idx == 0 else "HAPPY 😄"
            
        else:
            # ---> LOW ENERGY (Sad/Relaxed) - BERT
            branch = "Low Energy (Lyrics Focus)"
            text_clean = clean_text(lyrics_raw)
            
            if len(text_clean) < 5:
                mood = "UNKNOWN (No Lyrics)"
                conf = 0.0
            else:
                inputs = tokenizer(text_clean, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
                # Mapping id2label = {0: 'sad', 1: 'relaxed'}
                p_sad = probs[0].item()
                p_rel = probs[1].item()
                
                # --- LOGIKA SENTIMENTAL (AMBIGUITY) ---
                diff = abs(p_sad - p_rel)
                
                if diff < 0.15: # Jika beda tipis (< 15%)
                    mood = "SENTIMENTAL (Sad & Relaxed) 🍂"
                    conf = (p_sad + p_rel) / 2
                elif p_sad > p_rel:
                    mood = "SAD 😢"
                    conf = p_sad
                else:
                    mood = "RELAXED 😌"
                    conf = p_rel

        # Output Result
        bar_len = int(conf * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"📍 Route  : {branch}")
        print(f"🏆 Mood   : {mood}")
        print(f"🎯 Conf   : {conf*100:.1f}% |{bar}|")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error: {e}")

# ================= 5. RUNNER =================

if __name__ == "__main__":
    if not os.path.exists(AUDIO_FOLDER):
        print(f"⚠️ Folder '{AUDIO_FOLDER}' tidak ditemukan. Buat dulu folder ini dan isi lagu.")
        exit()

    # Loop semua lagu yang ada di Dictionary MANUAL_DATA
    found_any = False
    for fname, lirik in MANUAL_DATA.items():
        predict_song(fname, lirik)
        found_any = True
        
    if not found_any:
        print("⚠️ Dictionary MANUAL_DATA kosong atau tidak ada file yang cocok.")