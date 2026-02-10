import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- KONFIGURASI BARU (RAW 2) ---
RAW_DATA_DIR = 'data/raw2'  # <--- FOKUS KE SINI
FOLDS = 5
SEED = 42
TARGET_SR = 16000

print(f"🚀 MEMULAI EXP 13: RAW 2 DATASET (STAGE 1 + AUGMENTASI)...")
print(f"📂 Target Folder: {RAW_DATA_DIR}")

# --- 1. LOAD YAMNET ---
print("⏳ Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# --- 2. HELPER FUNCTIONS ---
def trim_middle(y, sr, percentage=0.5):
    """Potong 50% Tengah"""
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def extract_features_from_audio(y, sr=16000):
    """Ekstrak fitur langsung dari array audio"""
    try:
        # 1. YAMNet Embedding
        if np.max(np.abs(y)) > 0: y_norm = y / np.max(np.abs(y))
        else: y_norm = y
            
        if len(y_norm) < 16000: y_norm = np.pad(y_norm, (0, 16000 - len(y_norm)))
            
        _, embeddings, _ = yamnet_model(y_norm)
        
        mean = tf.reduce_mean(embeddings, axis=0)
        std = tf.math.reduce_std(embeddings, axis=0)
        max_ = tf.reduce_max(embeddings, axis=0)
        yamnet_emb = tf.concat([mean, std, max_], axis=0).numpy()
        
        # 2. Physics Features (RMS + ZCR) untuk deteksi energi
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        return np.concatenate([yamnet_emb, [rms, zcr]])
    except:
        return None

def augment_audio(y):
    """Membuat 2 variasi baru: Noise & Shift"""
    # 1. Noise Injection
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])
    
    # 2. Time Shift
    shift_len = int(TARGET_SR * 0.5) # Geser max 0.5 detik
    direction = np.random.choice([True, False])
    if direction:
        y_shift = np.roll(y, shift_len)
    else:
        y_shift = np.roll(y, -shift_len)
        
    return [y_noise, y_shift]

# --- 3. LOAD RAW AUDIO TO MEMORY ---
print("📥 Loading Raw 2 Audio ke Memory...")

audio_data = [] 
labels = []
filenames = [] # Debugging

moods = ['angry', 'happy', 'sad', 'relaxed']
found_files = 0

for mood in moods:
    folder = os.path.join(RAW_DATA_DIR, mood)
    if not os.path.exists(folder): 
        print(f"⚠️ Warning: Folder {folder} tidak ditemukan!")
        continue
        
    files = os.listdir(folder)
    for f in tqdm(files, desc=f"Load {mood}"):
        if not f.endswith(('.wav', '.mp3')): continue
        
        # Mapping Binary (High vs Low)
        binary_label = 'high' if mood in ['angry', 'happy'] else 'low'
        
        try:
            path = os.path.join(folder, f)
            y, sr = librosa.load(path, sr=TARGET_SR)
            y = trim_middle(y, sr) # Potong tengah
            
            audio_data.append(y)
            labels.append(binary_label)
            filenames.append(f)
            found_files += 1
        except: continue

if found_files == 0:
    print("❌ ERROR: Tidak ada file audio ditemukan di data/raw2!")
    exit()

audio_data = np.array(audio_data, dtype=object)
labels = np.array(labels)

print(f"✅ Loaded {len(audio_data)} audio files from RAW 2.")
print(f"   High Arousal: {np.sum(labels == 'high')}")
print(f"   Low Arousal : {np.sum(labels == 'low')}")

# --- 4. MODEL DEFINITION ---
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

# --- 5. K-FOLD WITH AUGMENTATION ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

fold_scores = []
confusion_matrices = []

# Encode Label
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

for fold, (train_idx, test_idx) in enumerate(skf.split(audio_data, y_encoded)):
    print(f"\n🔄 --- FOLD {fold+1} / {FOLDS} ---")
    
    # A. PREPARE TRAINING DATA (WITH AUGMENTATION)
    print("   🔨 Augmenting (3x)...")
    X_train_vec = []
    y_train_vec = []
    
    for i in tqdm(train_idx, leave=False):
        y_raw = audio_data[i]
        label_vec = y_categorical[i]
        
        # 1. Data Asli
        vec = extract_features_from_audio(y_raw)
        if vec is not None:
            X_train_vec.append(vec)
            y_train_vec.append(label_vec)
            
        # 2. Data Augmentasi
        variations = augment_audio(y_raw)
        for y_aug in variations:
            vec_aug = extract_features_from_audio(y_aug)
            if vec_aug is not None:
                X_train_vec.append(vec_aug)
                y_train_vec.append(label_vec)
    
    X_train_final = np.array(X_train_vec)
    y_train_final = np.array(y_train_vec)
    
    # B. PREPARE TEST DATA (PURE)
    print("   🧪 Extracting Test Data...")
    X_test_vec = []
    y_test_vec = []
    
    for i in test_idx:
        vec = extract_features_from_audio(audio_data[i])
        if vec is not None:
            X_test_vec.append(vec)
            y_test_vec.append(y_categorical[i])
            
    X_test_final = np.array(X_test_vec)
    y_test_final = np.array(y_test_vec)
    
    print(f"   📊 Train Size: {len(X_train_final)} | Test Size: {len(X_test_final)}")
    
    # C. TRAINING
    y_train_int = np.argmax(y_train_final, axis=1)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
    weights_dict = dict(enumerate(weights))
    
    model = create_model(X_train_final.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    model.fit(X_train_final, y_train_final,
              validation_data=(X_test_final, y_test_final),
              epochs=60, batch_size=32,
              callbacks=[early_stop], class_weight=weights_dict, verbose=0)
    
    # D. EVALUASI
    y_pred = np.argmax(model.predict(X_test_final, verbose=0), axis=1)
    y_true = np.argmax(y_test_final, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    fold_scores.append(acc)
    print(f"   🎯 Akurasi Fold {fold+1}: {acc*100:.2f}%")
    
    confusion_matrices.append(confusion_matrix(y_true, y_pred))

# --- 6. FINAL REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR RAW 2 (AUGMENTED)")
print("="*50)

avg_acc = np.mean(fold_scores) * 100
std_acc = np.std(fold_scores) * 100

print(f"📈 Rata-Rata Akurasi: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
print(f"📝 Per Fold: {[f'{x*100:.2f}%' for x in fold_scores]}")

# Plot Average CM
avg_cm = np.mean(confusion_matrices, axis=0).astype(int)
labels = le.classes_ 

plt.figure(figsize=(6,5))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
plt.title(f'Binary Arousal RAW 2 (Augmented)\nAvg Acc: {avg_acc:.2f}%')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.savefig('cm_exp13_raw2_augmented.png')
plt.show()