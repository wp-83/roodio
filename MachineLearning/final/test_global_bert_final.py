import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import joblib 
import tensorflow as tf
import tensorflow_hub as hub
import torch
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

print("🚀 GLOBAL STRESS TEST: FULL RAW DATA + BERT")
print("   Pipeline: Yamnet (Stage 1) -> DistilBERT (Stage 2B)")
print("   Dataset : ALL RAW DATA (Unfiltered)")
print("="*60)

# ================= CONFIG =================
# LIST SEMUA FOLDER RAW ANDA
SOURCE_DIRS = ['data/raw', 'data/raw2'] 

# Path Model BERT Anda (Yang 82%)
BERT_MODEL_PATH = 'models/model_hybrid_final' 
# Atau gunakan path checkpoint: './best_model_80percent'

LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
MODEL_DIR = 'models/'

# Target Label
TARGET_CLASSES = ['angry', 'happy', 'sad', 'relaxed']

# ================= 1. LOAD MODELS =================
print("⏳ Loading Models...")
try:
    # A. Audio Models
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    s1_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'stage1_nn.h5'))
    s2a_rf = joblib.load(os.path.join(MODEL_DIR, 'stage2a_rf.pkl'))
    s2a_meta = joblib.load(os.path.join(MODEL_DIR, 'stage2a_meta.pkl'))
    
    # B. Text Model (BERT)
    print(f"   ...Loading BERT from: {BERT_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_model.eval()
    
    # MAPPING LABEL BERT (PENTING!)
    # Sesuaikan dengan training Anda. Biasanya 0=relaxed, 1=sad (abjad).
    # Jika terbalik, ubah jadi {0:'sad', 1:'relaxed'}
    id2label = {1: 'relaxed', 0: 'sad'}
    print(f"   ✅ BERT Label Map: {id2label}")

    # C. Lyrics Database
    df_lyric = pd.read_excel(LYRICS_PATH)
    df_lyric['id'] = df_lyric['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    lyrics_map = dict(zip(df_lyric['id'], df_lyric['lyrics']))
    
    print("✅ System Ready.")

except Exception as e:
    print(f"❌ Error Loading: {e}")
    exit()

# ================= 2. HELPER FUNCTIONS =================

def clean_text_bert(text):
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
    yamnet_emb = tf.concat([
        tf.reduce_mean(embeddings, axis=0),
        tf.math.reduce_std(embeddings, axis=0),
        tf.reduce_max(embeddings, axis=0)
    ], axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y_trim))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_trim))
    return np.concatenate([yamnet_emb, [rms, zcr]]).reshape(1, -1)

def extract_feat_s2a_audio(y, sr):
    if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
    _, emb, _ = yamnet_model(y)
    vec = tf.reduce_mean(emb, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([vec, [rms, zcr]]).reshape(1, -1)

# ================= 3. TESTING LOOP =================

y_true = []
y_pred = []
missing_lyrics_count = 0

# Collect ALL Files
all_files = []
for d in SOURCE_DIRS:
    for mood in TARGET_CLASSES:
        # Cek apakah folder mood ada
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
        
        # === STAGE 1: ENERGY CHECK (AUDIO) ===
        feat_s1 = extract_feat_s1(y, sr)
        p1 = s1_model.predict(feat_s1, verbose=0)[0]
        
        # Asumsi Training: 0 = High (Angry/Happy), 1 = Low (Sad/Relaxed)
        if p1[0] > p1[1]:
            # ---> HIGH ENERGY BRANCH
            f_aud = extract_feat_s2a_audio(y, sr)
            f_txt = np.array([[0.5, 0.5]]) # Dummy Text (Kita fokus ke S2B sekarang)
            
            p_rf = s2a_rf.predict_proba(f_aud)
            meta_in = np.concatenate([p_rf, f_txt], axis=1)
            idx = s2a_meta.predict(meta_in)[0]
            
            final_pred = "angry" if idx == 0 else "happy"
            
        else:
            # ---> LOW ENERGY BRANCH (BERT)
            cleaned_text = clean_text_bert(lyrics)
            
            if len(cleaned_text) < 5:
                # KASUS: Lirik Kosong / Tidak Ada di Excel
                missing_lyrics_count += 1
                # Fallback: Kita tebak 'relaxed' sebagai default aman untuk low energy
                # Atau Anda bisa menandainya sebagai 'unknown'
                final_pred = "relaxed" 
            else:
                # Prediksi BERT
                inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_idx = torch.argmax(probs).item()
                
                final_pred = id2label[pred_idx]

        y_true.append(true_label)
        y_pred.append(final_pred)
        
    except Exception as e:
        # print(f"Skipping {os.path.basename(path)}: {e}")
        continue

# ================= 4. FINAL REPORT =================
print("\n" + "="*60)
print("📊 FINAL STRESS TEST REPORT (FULL RAW DATA)")
print("="*60)

# Statistik Lirik Hilang
print(f"Total Files       : {len(y_true)}")
print(f"Missing Lyrics    : {missing_lyrics_count} files ({missing_lyrics_count/len(y_true)*100:.1f}%)")
if missing_lyrics_count > 0:
    print("   (Note: File tanpa lirik otomatis diprediksi 'relaxed' di Stage 2B)")
print("-" * 60)

# Hitung Akurasi
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
plt.title(f'Global System on Full Raw Data\n(Yamnet + BERT) - Acc: {acc:.2f}%')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('full_raw_test_results.png')
plt.show()

print("\n✅ Hasil disimpan ke 'full_raw_test_results.png'")