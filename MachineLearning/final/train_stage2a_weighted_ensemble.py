import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import joblib 
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['angry', 'happy'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 30: WEIGHTED ENSEMBLE OPTIMIZATION...")
print(f"   (Mencari Keseimbangan Sempurna antara Audio & Teks)")

# ================= 1. PREPARE DATA =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    lyrics_map = pd.Series(df.lyrics.values, index=df.id).to_dict()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Loaded: {len(df)}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

def clean_lyrics_text(text):
    if pd.isna(text) or text == '': return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text) 
    text = re.sub(r'\(.*?\)', ' ', text) 
    garbage = ['lyrics', 'embed', 'contributors', 'translation']
    for w in garbage: text = text.replace(w, '')
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for pid in lyrics_map: lyrics_map[pid] = clean_lyrics_text(lyrics_map[pid])

print("⏳ Loading Extractors...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

X_audio_features = []
X_text_probs = [] # Kita simpan Probability Teks disini
y_labels = []

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, [rms, zcr]])
    except: return None

def get_text_prob_angry_happy(lyrics):
    # Output: [Prob_Angry, Prob_Happy]
    try:
        lyrics_chunk = str(lyrics)[:512]
        output = nlp_classifier(lyrics_chunk)[0]
        scores = {item['label']: item['score'] for item in output}
        
        # Agregasi skor
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
        
        # Normalisasi agar total = 1.0 (Probability Distribution)
        total = s_happy + s_angry + 1e-9
        return [s_angry/total, s_happy/total] 
    except: return [0.5, 0.5]

def get_id(path):
    base = os.path.basename(path)
    return base.split('_')[0].strip() if '_' in base else None

all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

print("🧠 Extracting Features...")
for path in tqdm(all_files):
    fid = get_id(path)
    if fid not in lyrics_map: continue
    
    mood = mood_map[fid]
    label = 0 if mood == 'angry' else 1
    
    aud = extract_audio(path)
    txt_probs = get_text_prob_angry_happy(lyrics_map[fid])
    
    if aud is not None:
        X_audio_features.append(aud)
        X_text_probs.append(txt_probs) # Simpan probabilitas teks [p_angry, p_happy]
        y_labels.append(label)

X_audio = np.array(X_audio_features)
X_text = np.array(X_text_probs)
y = np.array(y_labels)

# ================= 2. GENERATE PREDICTIONS (CROSS-VALIDATION) =================
print(f"\n🚀 Generating Audio Predictions (CV)...")

# Kita gunakan Random Forest yang "Solid" (dari hasil Benchmark)
# Tidak perlu tuning aneh-aneh di dalam loop
rf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=None, 
    min_samples_leaf=2, # Sedikit regularization agar tidak overconfident
    random_state=SEED, 
    n_jobs=-1
)

# Mendapatkan probabilitas Audio murni dari Cross Validation
# Ini mensimulasikan performa Audio Model pada data yang belum pernah dilihat
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
y_pred_audio_proba = cross_val_predict(rf, X_audio, y, cv=skf, method='predict_proba')

# Text probability sudah kita dapatkan langsung dari RoBERTa (Pre-trained)
# X_text sudah berisi [prob_angry, prob_happy]

# ================= 3. FIND OPTIMAL WEIGHT =================
print(f"\n⚖️ Searching for Optimal Weight (Grid Search)...")

best_acc = 0
best_weight = 0.0 # Berapa persen kita percaya Audio?
best_y_pred = []

# Kita coba bobot Audio dari 0.0 sampai 1.0 dengan step 0.01
weights_to_try = np.linspace(0, 1, 101)
acc_history = []

for w in weights_to_try:
    # Rumus Ensemble: P_final = (w * P_audio) + ((1-w) * P_text)
    y_proba_weighted = (w * y_pred_audio_proba) + ((1 - w) * X_text)
    
    # Ambil kelas dengan probabilitas tertinggi
    y_pred_temp = np.argmax(y_proba_weighted, axis=1)
    
    acc = accuracy_score(y, y_pred_temp)
    acc_history.append(acc)
    
    if acc > best_acc:
        best_acc = acc
        best_weight = w
        best_y_pred = y_pred_temp

# ================= 4. REPORT =================
print("\n" + "="*50)
print(f"🏆 HASIL AKHIR (WEIGHTED ENSEMBLE)")
print("="*50)
print(f"✅ Akurasi Terbaik : {best_acc*100:.2f}%")
print(f"⚖️ Bobot Optimal   : Audio {best_weight*100:.0f}% + Text {(1-best_weight)*100:.0f}%")

print("-" * 50)
print(classification_report(y, best_y_pred, target_names=['angry', 'happy']))

# Plot Weight Search
plt.figure(figsize=(8, 4))
plt.plot(weights_to_try, acc_history, linewidth=2)
plt.axvline(best_weight, color='r', linestyle='--', label=f'Best: {best_weight:.2f}')
plt.xlabel('Bobot Audio (0 = Text Only, 1 = Audio Only)')
plt.ylabel('Akurasi')
plt.title(f'Optimal Weight Search (Max Acc: {best_acc*100:.2f}%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('weight_optimization.png')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Angry','Happy'], yticklabels=['Angry','Happy'])
plt.title(f'Weighted Ensemble\nAudio {best_weight:.2f} | Text {1-best_weight:.2f}')
plt.show()

# ================= 5. SAVE FINAL MODEL =================
print("\n💾 Saving Final Models...")
# 1. Train Audio Model Full
rf.fit(X_audio, y)
joblib.dump(rf, 'models/stage2a_audio_rf.pkl')

# 2. Simpan Info Bobot
with open('models/stage2a_ensemble_info.txt', 'w') as f:
    f.write(f"Ensemble Type: Weighted Soft Voting\n")
    f.write(f"Audio Model: RandomForest (n=300, min_leaf=2)\n")
    f.write(f"Text Model: RoBERTa (Zero-Shot)\n")
    f.write(f"Audio Weight: {best_weight:.4f}\n")
    f.write(f"Text Weight: {1-best_weight:.4f}\n")
    f.write(f"Accuracy: {best_acc*100:.2f}%\n")

print("✅ Model Audio & Konfigurasi Bobot Disimpan!")
print("   Saat prediksi nanti, gunakan rumus:")
print(f"   Final_Prob = ({best_weight:.2f} * Audio_Prob) + ({1-best_weight:.2f} * Text_Prob)")