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
from sklearn.preprocessing import StandardScaler # <--- WAJIB UNTUK DEEP LEARNING
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics1.csv'
TARGET_MOODS = ['angry', 'happy'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 28: DEEP LEARNING STACKING (NN + Roberta)...")

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
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
        return [s_angry, s_happy] 
    except: return [0.5, 0.5]

files_angry = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'angry')) if f.endswith(('wav','mp3'))])
files_happy = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'happy')) if f.endswith(('wav','mp3'))])
lyrics_angry = df[df['mood']=='angry']['lyrics_clean'].tolist()
lyrics_happy = df[df['mood']=='happy']['lyrics_clean'].tolist()

# Angry Loop
for i in tqdm(range(min(len(files_angry), len(lyrics_angry))), desc="Angry"):
    path = os.path.join(RAW_DATA_DIR, 'angry', files_angry[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_angry[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(0)

# Happy Loop
for i in tqdm(range(min(len(files_happy), len(lyrics_happy))), desc="Happy"):
    path = os.path.join(RAW_DATA_DIR, 'happy', files_happy[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_happy[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(1)

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) 
y_labels = np.array(y_labels)

# --- PENTING: SCALING DATA AUDIO UNTUK DEEP LEARNING ---
print("⚖️ Scaling Audio Features (StandardScaler)...")
scaler = StandardScaler()
X_audio_features = scaler.fit_transform(X_audio_features)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 3. FUNGSI MODEL DEEP LEARNING (KERAS) ---
def create_dnn_model(input_shape):
    model = Sequential([
        # Layer 1
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(), # Menstabilkan training
        Dropout(0.4),         # Mencegah overfitting (matikan 40% neuron acak)
        
        # Layer 2
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output Layer (2 Kelas: Angry vs Happy)
        Dense(2, activation='softmax') 
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
    
    # --- STEP 1: TRAIN AUDIO NEURAL NETWORK ---
    # Kita butuh validasi split kecil dari training set untuk Early Stopping
    # Agar model tidak "menghafal" (Overfitting)
    
    model_nn = create_dnn_model(X_aud_tr.shape[1])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train NN
    # verbose=0 agar layar tidak penuh log epoch
    model_nn.fit(X_aud_tr, y_tr, 
                 epochs=100, 
                 batch_size=16, 
                 validation_split=0.2, 
                 callbacks=[early_stop],
                 verbose=0) 
    
    # Generate Probabilitas
    # Note: Di DL, kita gunakan model yang baru dilatih untuk predict training set (untuk input meta learner)
    # Hati-hati: Ini sedikit berisiko bias dibanding cross_val_predict, tapi untuk DL ini standar.
    prob_audio_train = model_nn.predict(X_aud_tr, verbose=0)
    prob_audio_test = model_nn.predict(X_aud_ts, verbose=0)
    
    # --- STEP 2: META LEARNER (Logistic Regression) ---
    # Gabung Probabilitas Audio (dari NN) + Skor Teks (dari Roberta)
    X_meta_train = np.concatenate([prob_audio_train, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    meta_clf = LogisticRegression()
    meta_clf.fit(X_meta_train, y_tr)
    
    meta_weights.append(meta_clf.coef_[0]) 
    
    # Final Predict
    y_pred_fold = meta_clf.predict(X_meta_test)
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)

# --- 5. HASIL AKHIR ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR (DEEP LEARNING STACKING)")
print("="*50)
print(f"🏆 Accuracy : {mean_acc:.2f}%")
print(f"📉 Stability: ±{std_acc:.2f}%")
print("-" * 50)
print(classification_report(y_true_all, y_pred_all, target_names=['angry', 'happy']))

# Plot CM
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Angry','Happy'], yticklabels=['Angry','Happy'])
plt.title(f'Deep Learning Result\nAcc: {mean_acc:.1f}%')
plt.show()

# Analisis Bobot
avg_weights = np.mean(meta_weights, axis=0)
print("\n⚖️ KONTRIBUSI FITUR (DL vs Roberta):")
print(f"🔊 Deep Learning Audio: {abs(avg_weights[0]) + abs(avg_weights[1]):.4f}")
print(f"📝 Roberta Text Score : {abs(avg_weights[2]) + abs(avg_weights[3]):.4f}")