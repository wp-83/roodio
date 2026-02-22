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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from transformers import pipeline
from sklearn.svm import SVC  # <--- GANTI META LEARNER KE SVM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler 
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
RAW_DATA_DIR = 'data/raw'
LYRICS_PATH = 'data/lyrics/lyrics1.csv'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 42 # Coba ganti seed sedikit

print(f"🚀 MEMULAI EXP 29: TUNING AUGMENTATION + SVM (SAD vs RELAXED)...")

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

# Load Models
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
        
        # Tambahan fitur statistik biar lebih kaya
        yamnet_std = tf.math.reduce_std(emb, axis=0).numpy()
        
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # Gabung Mean + Std + RMS + ZCR
        return np.concatenate([yamnet_vec, yamnet_std, [rms, zcr]])
    except: return None

def get_text_scores(lyrics):
    try:
        lyrics_chunk = str(lyrics)[:512]
        output = nlp_classifier(lyrics_chunk)[0]
        scores = {item['label']: item['score'] for item in output}
        
        s_relaxed = scores.get('neutral', 0) + scores.get('joy', 0) + scores.get('surprise', 0)
        s_sad = scores.get('sadness', 0) + scores.get('fear', 0) + scores.get('anger', 0) + scores.get('disgust', 0)
        
        return [s_sad, s_relaxed] 
    except: return [0.5, 0.5]

files_sad = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'sad')) if f.endswith(('wav','mp3'))])
files_relaxed = sorted([f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'relaxed')) if f.endswith(('wav','mp3'))])
lyrics_sad = df[df['mood']=='sad']['lyrics_clean'].tolist()
lyrics_relaxed = df[df['mood']=='relaxed']['lyrics_clean'].tolist()

for i in tqdm(range(min(len(files_sad), len(lyrics_sad))), desc="Sad"):
    path = os.path.join(RAW_DATA_DIR, 'sad', files_sad[i])
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_sad[i])
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(0)

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

# Scaling Audio
print("⚖️ Scaling Audio Features...")
scaler = StandardScaler()
X_audio_features = scaler.fit_transform(X_audio_features)

# --- FUNGSI AUGMENTASI DATA SINTETIS ---
def augment_features(X, y, noise_level=0.05, copies=3):
    X_aug, y_aug = [], []
    for _ in range(copies):
        noise = np.random.normal(0, noise_level, X.shape)
        X_new = X + noise
        X_aug.append(X_new)
        y_aug.append(y)
    
    # Gabung data asli + data noise
    return np.vstack([X] + X_aug), np.concatenate([y] + y_aug)

# --- MODEL NEURAL NETWORK (SIMPLIFIED) ---
def create_tuned_model(input_shape):
    model = Sequential([
        # Kita kurangi jumlah neuron agar tidak overfitting
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.5), # Dropout tinggi (50%) karena data sedikit
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(2, activation='softmax') 
    ])
    
    # Learning rate lebih kecil biar belajar pelan tapi pasti
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- TRAINING LOOP ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []

print(f"\n🚀 START TUNING (Augmentation x3 + SVM Meta)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # 1. AUGMENTASI DATA TRAINING (Audio Saja)
    # Kita perbanyak data training 3x lipat dengan noise
    X_aud_tr_aug, y_tr_aug = augment_features(X_aud_tr, y_tr, copies=3)
    
    # 2. TRAIN NEURAL NETWORK
    model_nn = create_tuned_model(X_aud_tr.shape[1])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    model_nn.fit(X_aud_tr_aug, y_tr_aug, # Pakai data augmentasi
                 epochs=200, 
                 batch_size=16, 
                 validation_split=0.2, 
                 callbacks=[early_stop, lr_schedule],
                 verbose=0)
    
    # 3. PREDIKSI (Pakai data asli, bukan augmented)
    prob_audio_train = model_nn.predict(X_aud_tr, verbose=0)
    prob_audio_test = model_nn.predict(X_aud_ts, verbose=0)
    
    # 4. META LEARNER: SVM (Support Vector Machine)
    # SVM lebih jago memisahkan margin data yang mepet dibanding Logistic Regression
    X_meta_train = np.concatenate([prob_audio_train, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # Tuning SVM: C=1.0 (Regularization), Kernel RBF (Non-linear)
    meta_clf = SVC(kernel='rbf', C=1.5, probability=True, random_state=SEED)
    meta_clf.fit(X_meta_train, y_tr)
    
    y_pred_fold = meta_clf.predict(X_meta_test)
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)

# --- REPORT ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR TUNING (AUGMENTASI + SVM)")
print("="*50)
print(f"🏆 Average Accuracy : {mean_acc:.2f}%")
print(f"📉 Stability (Std)  : ±{std_acc:.2f}%")
print("-" * 50)
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Tuned Result\nAcc: {mean_acc:.1f}%')
plt.show()