import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import LeaveOneOut
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
SEED = 42
TARGET_SR = 16000

print(f"🚀 MEMULAI EXP 14: LOOCV (LEAVE-ONE-OUT) - UJIAN PALING JUJUR...")

# --- 1. LOAD YAMNET ---
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# --- 2. HELPER FUNCTIONS ---
def trim_middle(y, sr, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def extract_features_from_audio(y, sr=16000):
    try:
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
        return np.concatenate([yamnet_emb, [rms, zcr]])
    except: return None

def augment_audio(y):
    # Augmentasi Noise & Shift
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])
    
    shift_len = int(TARGET_SR * 0.5)
    direction = np.random.choice([True, False])
    y_shift = np.roll(y, shift_len) if direction else np.roll(y, -shift_len)
        
    return [y_noise, y_shift]

# --- 3. LOAD AUDIO ---
print("📥 Loading Raw 2 Audio...")
audio_data = [] 
labels = []
filenames = [] # Kita simpan nama file biar tau SIAPA PELAKU ERRORNYA

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
            filenames.append(f)
        except: continue

audio_data = np.array(audio_data, dtype=object)
labels = np.array(labels)
filenames = np.array(filenames)

print(f"✅ Total File: {len(filenames)}")

# --- 4. MODEL ---
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

# --- 5. LOOCV LOOP ---
loo = LeaveOneOut()
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

y_true_all = []
y_pred_all = []
errors = []

print(f"\n🚀 MULAI LOOCV ({len(filenames)} Iterasi)...")

# Loop sebanyak jumlah file
for i, (train_idx, test_idx) in enumerate(tqdm(loo.split(audio_data), total=len(filenames))):
    
    # 1. Augmentasi Training Data
    X_train_vec = []
    y_train_vec = []
    
    for idx in train_idx:
        y_raw = audio_data[idx]
        lbl = y_categorical[idx]
        
        # Asli
        vec = extract_features_from_audio(y_raw)
        if vec is not None:
            X_train_vec.append(vec)
            y_train_vec.append(lbl)
        
        # Augmentasi
        for y_aug in augment_audio(y_raw):
            vec_aug = extract_features_from_audio(y_aug)
            if vec_aug is not None:
                X_train_vec.append(vec_aug)
                y_train_vec.append(lbl)
                
    X_train_final = np.array(X_train_vec)
    y_train_final = np.array(y_train_vec)
    
    # 2. Test Data (Cuma 1 file)
    test_idx_val = test_idx[0]
    X_test_vec = extract_features_from_audio(audio_data[test_idx_val]).reshape(1, -1)
    y_test_vec = y_categorical[test_idx_val].reshape(1, -1)
    
    # 3. Train Model
    y_train_int = np.argmax(y_train_final, axis=1)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
    weights_dict = dict(enumerate(weights))
    
    model = create_model(X_train_final.shape[1])
    # Epoch dikit aja karena LOOCV lama, dan data dikit cepet converge
    model.fit(X_train_final, y_train_final, epochs=30, batch_size=32, 
              verbose=0, class_weight=weights_dict)
    
    # 4. Predict
    pred_prob = model.predict(X_test_vec, verbose=0)
    pred_idx = np.argmax(pred_prob)
    true_idx = np.argmax(y_test_vec)
    
    y_true_all.append(true_idx)
    y_pred_all.append(pred_idx)
    
    # Cek Error
    if pred_idx != true_idx:
        fname = filenames[test_idx_val]
        actual = le.inverse_transform([true_idx])[0]
        predicted = le.inverse_transform([pred_idx])[0]
        errors.append([fname, actual, predicted])

# --- 6. HASIL AKHIR ---
print("\n" + "="*50)
print("📊 HASIL AKHIR LOOCV")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Akurasi Total: {final_acc:.2f}%")

if len(errors) > 0:
    print("\n❌ DAFTAR FILE BIANG KEROK (ERROR):")
    print(f"{'Filename':<50} | {'True':<10} | {'Pred':<10}")
    print("-" * 75)
    for e in errors:
        print(f"{e[0]:<50} | {e[1]:<10} | {e[2]:<10}")
else:
    print("\n🎉 SEMPURNA! Tidak ada error.")

# Matrix
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'LOOCV Result (Raw 2)\nAccuracy: {final_acc:.2f}%')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig('cm_exp14_loocv.png')
plt.show()