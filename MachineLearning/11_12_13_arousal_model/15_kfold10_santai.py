import os
import logging

# --- 1. MODE SANTAI (Mute Warning TensorFlow) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- KONFIGURASI ---
RAW_DATA_DIR = 'data/raw2' 
FOLDS = 10  # <--- GANTI JADI 10
SEED = 42
TARGET_SR = 16000

print(f"🚀 MEMULAI EXP 15: 10-FOLD CV (AUGMENTED & SANTAI)...")

# --- LOAD YAMNET ---
print("⏳ Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# --- HELPER FUNCTIONS ---
def trim_middle(y, sr, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def extract_features_from_audio(y, sr=16000):
    try:
        # Cast ke float32 biar TF gak rewel
        y = y.astype(np.float32)

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
        
        return np.concatenate([yamnet_emb, [rms, zcr]]).astype(np.float32)
    except: return None

def augment_audio(y):
    # Augmentasi: Noise & Shift
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])
    
    shift_len = int(TARGET_SR * 0.5)
    direction = np.random.choice([True, False])
    y_shift = np.roll(y, shift_len) if direction else np.roll(y, -shift_len)
        
    return [y_noise.astype(np.float32), y_shift.astype(np.float32)]

# --- LOAD AUDIO ---
print("📥 Loading Raw 2 Audio...")
audio_data = [] 
labels = []

moods = ['angry', 'happy', 'sad', 'relaxed']
for mood in moods:
    folder = os.path.join(RAW_DATA_DIR, mood)
    if not os.path.exists(folder): continue
    for f in tqdm(os.listdir(folder), desc=f"Load {mood}"):
        if not f.endswith(('.wav', '.mp3')): continue
        path = os.path.join(folder, f)
        try:
            y, sr = librosa.load(path, sr=TARGET_SR)
            y = trim_middle(y, sr)
            audio_data.append(y)
            labels.append('high' if mood in ['angry', 'happy'] else 'low')
        except: continue

audio_data = np.array(audio_data, dtype=object)
labels = np.array(labels)

print(f"✅ Total File: {len(audio_data)}")
print(f"   High: {np.sum(labels=='high')} | Low: {np.sum(labels=='low')}")

# --- MODEL ---
def create_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 10-FOLD LOOP ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

fold_scores = []
confusion_matrices = []

print(f"\n🚀 MULAI 10-FOLD CV...")

for fold, (train_idx, test_idx) in enumerate(skf.split(audio_data, y_encoded)):
    print(f"\n🔄 FOLD {fold+1} / {FOLDS}")
    
    # 1. Prepare Training (With Augmentation)
    X_train_vec = []
    y_train_vec = []
    
    # Loop manual biar progress bar keliatan dikit
    # (Opsional: kalau terlalu flooding bisa dimatikan tqdmnnya)
    for idx in train_idx:
        y_raw = audio_data[idx]
        lbl = y_categorical[idx]
        
        # Asli
        vec = extract_features_from_audio(y_raw)
        if vec is not None:
            X_train_vec.append(vec)
            y_train_vec.append(lbl)
        
        # Augmentasi (Hanya Training!)
        for y_aug in augment_audio(y_raw):
            vec_aug = extract_features_from_audio(y_aug)
            if vec_aug is not None:
                X_train_vec.append(vec_aug)
                y_train_vec.append(lbl)
                
    X_train_final = np.array(X_train_vec, dtype=np.float32)
    y_train_final = np.array(y_train_vec, dtype=np.float32)
    
    # 2. Prepare Test (Pure)
    X_test_vec = []
    y_test_vec = []
    for idx in test_idx:
        vec = extract_features_from_audio(audio_data[idx])
        if vec is not None:
            X_test_vec.append(vec)
            y_test_vec.append(y_categorical[idx])
            
    X_test_final = np.array(X_test_vec, dtype=np.float32)
    y_test_final = np.array(y_test_vec, dtype=np.float32)
    
    # 3. Train
    y_train_int = np.argmax(y_train_final, axis=1)
    if len(np.unique(y_train_int)) > 1:
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
        weights_dict = dict(enumerate(weights))
    else: weights_dict = None
    
    model = create_model(X_train_final.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Silent Training
    model.fit(X_train_final, y_train_final, 
              validation_data=(X_test_final, y_test_final),
              epochs=50, batch_size=32, verbose=0, 
              callbacks=[early_stop], class_weight=weights_dict)
    
    # 4. Evaluate
    y_pred = np.argmax(model.predict(X_test_final, verbose=0), axis=1)
    y_true = np.argmax(y_test_final, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    fold_scores.append(acc)
    print(f"   🎯 Akurasi: {acc*100:.2f}%")
    
    confusion_matrices.append(confusion_matrix(y_true, y_pred))

# --- REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR 10-FOLD CV (RAW 2)")
print("="*50)

avg_acc = np.mean(fold_scores) * 100
std_acc = np.std(fold_scores) * 100

print(f"📈 Rata-Rata Akurasi: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
print(f"📝 Per Fold: {[f'{x*100:.2f}%' for x in fold_scores]}")

# Average CM
avg_cm = np.mean(confusion_matrices, axis=0).astype(int)
plt.figure(figsize=(6,5))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'10-Fold CV Result\nAvg Acc: {avg_acc:.2f}%')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig('cm_exp15_10fold.png')
plt.show()