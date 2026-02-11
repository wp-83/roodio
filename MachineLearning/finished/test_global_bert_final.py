import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import joblib 
import tensorflow as tf
import tensorflow_hub as hub # Masih butuh ini CUMA buat ekstraksi fitur YAMNet
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

print("🚀 GLOBAL STRESS TEST: FULL RAW DATA + PYTORCH ENGINE")
print("   Pipeline: Yamnet -> Stage 1 (PyTorch) -> Stage 2A (RF) / Stage 2B (BERT)")
print("   Dataset : ALL RAW DATA (Unfiltered)")
print("="*60)

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
BERT_MODEL_PATH = 'models/model_hybrid_final' # Path Model BERT
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
MODEL_DIR = 'models/v2' # Folder tempat stage1_nn.pth berada

TARGET_CLASSES = ['angry', 'happy', 'sad', 'relaxed']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running Inference on: {DEVICE}")

# ================= 1. DEFINISI ARSITEKTUR STAGE 1 (PYTORCH) =================
# PERBAIKAN: Menggunakan struktur layer terpisah (sesuai training awal)
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=3074):
        super(AudioClassifier, self).__init__()
        # Definisi Layer persis seperti saat Training
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
        self.output = nn.Linear(256, 2) # Output layer terpisah

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)

# ================= 2. LOAD MODELS =================
print("⏳ Loading Models...")
try:
    # A. Feature Extractor (YAMNet)
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # B. Stage 1: Audio Neural Network (PyTorch)
    s1_path = os.path.join(MODEL_DIR, 'stage1_nn.pth')
    s1_model = AudioClassifier().to(DEVICE)
    s1_model.load_state_dict(torch.load(s1_path, map_location=DEVICE))
    s1_model.eval()
    print("   ✅ Stage 1 (Audio) Loaded (PyTorch).")
    
    # Load Encoder Stage 1 (Untuk tahu 0 itu High atau Low)
    s1_encoder = joblib.load(os.path.join(MODEL_DIR, 'stage1_encoder.pkl'))
    
    # C. Stage 2A: Random Forest (Scikit-Learn)
    # Pastikan path ini benar (kadang di 'models/' atau 'models/v2/')
    s2a_rf = joblib.load(os.path.join('models', 'stage2a_rf.pkl'))
    s2a_meta = joblib.load(os.path.join('models', 'stage2a_meta.pkl'))
    
    # D. Stage 2B: Text Model (BERT)
    print(f"   ...Loading BERT from: {BERT_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH).to(DEVICE)
    bert_model.eval()
    
    # MAPPING LABEL BERT 
    # Cek config.json di folder model BERT Anda untuk memastikan id2label
    # Default asumsi: 0=Sad, 1=Relaxed (berdasarkan abjad training biasanya)
    # Jika hasil Sad/Relaxed terbalik, tukar dictionary ini.
    id2label_bert = {0: 'sad', 1: 'relaxed'} 
    print(f"   ✅ BERT Label Map: {id2label_bert}")

    # E. Lyrics Database
    df_lyric = pd.read_excel(LYRICS_PATH)
    df_lyric['id'] = df_lyric['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    lyrics_map = dict(zip(df_lyric['id'], df_lyric['lyrics']))
    
    print("✅ System Ready.")

except Exception as e:
    print(f"❌ Error Loading Models: {e}")
    exit()

# ================= 3. HELPER FUNCTIONS =================

def clean_text_bert(text):
    text = re.sub(r'\[.*?\]', '', str(text))
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def extract_feat_s1(y, sr=16000):
    # Ekstraksi Fitur untuk Stage 1 (YAMNet + RMS + ZCR)
    if len(y) < sr: y = np.pad(y, (0, sr - len(y)))
    # Trim tengah 50%
    start = int(len(y) * 0.25)
    end = start + int(len(y) * 0.5)
    y_trim = y[start:end]
    
    # YAMNet Embedding (Pakai y penuh atau trim tergantung training, disini pakai trim biar konsisten)
    _, embeddings, _ = yamnet_model(y_trim)
    
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_std  = tf.math.reduce_std(embeddings, axis=0).numpy()
    emb_max  = tf.reduce_max(embeddings, axis=0).numpy()
    
    rms = np.mean(librosa.feature.rms(y=y_trim))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_trim))
    
    return np.concatenate([emb_mean, emb_std, emb_max, [rms, zcr]])

def extract_feat_rf(y, sr=16000):
    # Fitur khusus Random Forest (Biasanya Yamnet Mean + RMS + ZCR saja)
    # Sesuaikan dengan training Stage 2A Anda
    if len(y) < sr: y = np.pad(y, (0, sr - len(y)))
    _, embeddings, _ = yamnet_model(y)
    
    vec = tf.reduce_mean(embeddings, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([vec, [rms, zcr]])

# ================= 4. TESTING LOOP =================

y_true = []
y_pred = []
missing_lyrics_count = 0

# Collect ALL Files
all_files = []
for d in SOURCE_DIRS:
    for mood in TARGET_CLASSES:
        mood_path = os.path.join(d, mood)
        if os.path.exists(mood_path):
            files = glob.glob(os.path.join(mood_path, '*.wav')) + \
                    glob.glob(os.path.join(mood_path, '*.mp3'))
            for f in files:
                all_files.append((f, mood))

print(f"\n📂 Processing {len(all_files)} files from RAW DATASETS...")

for path, true_label in tqdm(all_files):
    try:
        fid = os.path.basename(path).split('_')[0]
        lyrics = lyrics_map.get(fid, "")
        
        # Load Audio
        y, sr = librosa.load(path, sr=16000)
        
        # === STAGE 1: ENERGY CHECK (PYTORCH) ===
        feat_s1 = extract_feat_s1(y, sr)
        feat_tensor = torch.tensor([feat_s1], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            logits = s1_model(feat_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            
        # Decode label (0/1 -> 'high'/'low' or 'happy'/'sad' depending on encoder)
        # Kita asumsikan encoder mapping: High Energy vs Low Energy
        stage1_label = s1_encoder.inverse_transform([pred_idx])[0].lower()
        
        # Cek apakah High Energy (Angry/Happy)
        is_high_energy = False
        if 'high' in stage1_label or 'angry' in stage1_label or 'happy' in stage1_label:
            is_high_energy = True
        
        if is_high_energy:
            # ---> HIGH ENERGY BRANCH (RF)
            f_rf = extract_feat_rf(y, sr).reshape(1, -1)
            f_txt_dummy = np.array([[0.5, 0.5]]) # Dummy Text Feature
            
            p_rf = s2a_rf.predict_proba(f_rf)
            meta_in = np.concatenate([p_rf, f_txt_dummy], axis=1)
            idx = s2a_meta.predict(meta_in)[0] # 0=Angry, 1=Happy (Cek training RF!)
            
            # Mapping manual hasil RF (Sesuaikan dengan training Anda)
            final_pred = "angry" if idx == 0 else "happy"
            
        else:
            # ---> LOW ENERGY BRANCH (BERT)
            cleaned_text = clean_text_bert(lyrics)
            
            if len(cleaned_text) < 5:
                missing_lyrics_count += 1
                final_pred = "relaxed" # Fallback
            else:
                inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_bert_idx = torch.argmax(probs).item()
                
                final_pred = id2label_bert[pred_bert_idx]

        y_true.append(true_label)
        y_pred.append(final_pred)
        
    except Exception as e:
        # print(f"Skipping {os.path.basename(path)}: {e}")
        continue

# ================= 5. FINAL REPORT =================
print("\n" + "="*60)
print("📊 FINAL STRESS TEST REPORT (FULL RAW DATA)")
print("="*60)

print(f"Total Files       : {len(y_true)}")
print(f"Missing Lyrics    : {missing_lyrics_count} files ({missing_lyrics_count/len(y_true)*100:.1f}%)")
if missing_lyrics_count > 0:
    print("   (Note: File low energy tanpa lirik otomatis diprediksi 'relaxed')")
print("-" * 60)

labels_present = sorted(list(set(y_true + y_pred)))
acc = accuracy_score(y_true, y_pred) * 100

print(f"🏆 GLOBAL ACCURACY: {acc:.2f}%")
print("-" * 60)
print(classification_report(y_true, y_pred, labels=labels_present))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels_present)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu', 
            xticklabels=labels_present, yticklabels=labels_present)
plt.title(f'Global System (PyTorch + BERT)\nAcc: {acc:.2f}%')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('full_raw_pytorch_test.png')
plt.show()

print("\n✅ Hasil disimpan ke 'full_raw_pytorch_test.png'")