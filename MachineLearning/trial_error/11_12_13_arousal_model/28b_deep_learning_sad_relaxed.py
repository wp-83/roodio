import os
import re
import logging

# Mute Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler 
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI KHUSUS SAD/RELAXED ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics1.csv'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 28-B: DEEP LEARNING STACKING (SAD vs RELAXED)...")

# --- 1. SETUP & CLEANING DATA ---
try:
    df = pd.read_csv(LYRICS_PATH, sep=';', engine='python')
    if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',', engine='python')
except FileNotFoundError:
    print(f"❌ File {LYRICS_PATH} tidak ditemukan.")
    exit()

df.columns = df.columns.str.strip().str.lower()
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

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

df['lyrics_clean'] = df['lyrics'].apply(clean_lyrics_text)
print("✅ Lirik berhasil dibersihkan.")

# Load Feature Extractors
print("⏳ Loading Models (YAMNet & RoBERTa)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 2. FEATURE EXTRACTION ---
X_audio_features = []
X_text_scores = [] 
y_labels = []

print("🧠 Extracting Features...")

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

def get_text_scores(lyrics):
    try:
        lyrics_chunk = str(lyrics)[:512]
        output = nlp_classifier(lyrics_chunk)[0]
        scores = {item['label']: item['score'] for item in output}
        
        # LOGIKA SAD vs RELAXED
        # Relaxed = Neutral + Joy + Surprise
        s_relaxed = scores.get('neutral', 0) + scores.get('joy', 0) + scores.get('surprise', 0)
        # Sad = Sadness + Fear + Anger + Disgust
        s_sad = scores.get('sadness', 0) + scores.get('fear', 0) + scores.get('anger', 0) + scores.get('disgust', 0)
        
        return [s_sad, s_relaxed] 
    except: return [0.5, 0.5]

files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])
lyrics_sad = df[df['mood']=='sad']['lyrics_clean'].tolist()
lyrics_relaxed = df[df['mood']=='relaxed']['lyrics_clean'].tolist()

# Sad Loop (Label 0)
for i in tqdm(range(min(len(files_sad), len(lyrics_sad))), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_sad[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(0)

# Relaxed Loop (Label 1)
for i in tqdm(range(min(len(files_relaxed), len(lyrics_relaxed))), desc="Relaxed"):
    path = os.path.join(RAW_DATA_DIR, 'relaxed', files_relaxed[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_relaxed[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(1)

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) 
y_labels = np.array(y_labels)

# --- PENTING: SCALING AUDIO (Deep Learning butuh ini) ---
print("⚖️ Scaling Audio Features (StandardScaler)...")
scaler = StandardScaler()
X_audio_features = scaler.fit_transform(X_audio_features)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 3. BUILD DEEP LEARNING MODEL (Neural Network) ---
def create_dnn_model(input_shape):
    model = Sequential([
        # Hidden Layer 1: Besar & Kuat
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(), # Agar stabil
        Dropout(0.5),         # Mencegah overfit (penting karena data sedikit)
        
        # Hidden Layer 2: Menyaring Fitur
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden Layer 3: Penentu Akhir
        Dense(64, activation='relu'),
        
        # Output: 2 Neuron (Sad, Relaxed)
        Dense(2, activation='softmax') 
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Learning rate lambat agar teliti
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 4. STACKING TRAINING LOOP ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
meta_weights = []

print(f"\n🚀 START TRAINING (Neural Network + Stacking)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    # Split Data
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # --- LEVEL 1: TRAIN AUDIO NEURAL NETWORK ---
    model_nn = create_dnn_model(X_aud_tr.shape[1])
    
    # Stop kalau loss validasi gak turun-turun selama 15 epoch
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Training NN (Pakai 20% data training sebagai validasi internal)
    model_nn.fit(X_aud_tr, y_tr, 
                 epochs=150, 
                 batch_size=16, 
                 validation_split=0.2, 
                 callbacks=[early_stop],
                 verbose=0) # Silent mode
    
    # Generate Probabilitas untuk Meta Learner
    # Predict Train (untuk input training meta)
    prob_audio_train = model_nn.predict(X_aud_tr, verbose=0)
    # Predict Test (untuk input testing meta)
    prob_audio_test = model_nn.predict(X_aud_ts, verbose=0)
    
    # --- LEVEL 2: META LEARNER (Logistic Regression) ---
    # Gabung: [Prob_Audio_Sad, Prob_Audio_Relaxed, Score_Text_Sad, Score_Text_Relaxed]
    X_meta_train = np.concatenate([prob_audio_train, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    meta_clf = LogisticRegression()
    meta_clf.fit(X_meta_train, y_tr)
    
    meta_weights.append(meta_clf.coef_[0]) 
    
    # Evaluasi Akhir
    y_pred_fold = meta_clf.predict(X_meta_test)
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    
    print(f"   Fold {fold+1}: {acc*100:.2f}% (NN Audio Accuracy)")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)

# --- 5. HASIL AKHIR ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (DEEP LEARNING SAD vs RELAXED)")
print("="*50)
print(f"🏆 Average Accuracy : {mean_acc:.2f}%")
print(f"📉 Stability (Std)  : ±{std_acc:.2f}%")

if std_acc < 5.0: print("   ✅ SANGAT STABIL")
elif std_acc < 10.0: print("   ⚠️ CUKUP STABIL")
else: print("   ❌ KURANG STABIL")

print("-" * 50)
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

# Plot CM
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Deep Learning Result\nAcc: {mean_acc:.1f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('cm_exp28b_sad_relaxed.png')
plt.show()

# Analisis Bobot
avg_weights = np.mean(meta_weights, axis=0)
# Urutan: [Audio_Sad, Audio_Rel, Text_Sad, Text_Rel]
aud_score = abs(avg_weights[0]) + abs(avg_weights[1])
txt_score = abs(avg_weights[2]) + abs(avg_weights[3])

print("\n⚖️ KONTRIBUSI FITUR (NN vs RoBERTa):")
print(f"🔊 Deep Learning Audio: {aud_score:.4f}")
print(f"📝 Roberta Text Score : {txt_score:.4f}")
print(f"👉 Dominasi: {'Audio (NN)' if aud_score > txt_score else 'Lirik (Text)'}")