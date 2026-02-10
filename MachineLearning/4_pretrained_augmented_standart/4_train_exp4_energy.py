import os
import random
import numpy as np
import tensorflow as tf

# --- 1. KUNCI RANDOMNESS (SEED) ---
def set_seed(seed=34):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"🔒 Random Seed dikunci di angka: {seed}")

set_seed(34) 

# --- IMPORT LIBRARY ---
import tensorflow_hub as hub
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import mlflow
import mlflow.tensorflow
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]
DATA_DIR = PROJECT_DIR / 'data' / 'processed_exp4' 
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_exp4_energy'
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000 

def load_yamnet(): 
    return hub.load(YAMNET_URL)

# --- 2. ENERGY-AWARE FEATURE EXTRACTION ---
def extract_energy_features(yamnet, file_path):
    try:
        # Load Audio
        wav, _ = librosa.load(file_path, sr=SR)
        if np.max(np.abs(wav)) > 0: 
            wav = wav / np.max(np.abs(wav))
        
        if len(wav) < 15600: 
            wav = np.pad(wav, (0, 15600 - len(wav)))
            
        # 1. Dapatkan Embeddings dari YAMNet
        # YAMNet output: (N_frames, 1024)
        _, embeddings, _ = yamnet(wav)
        embeddings = embeddings.numpy()
        n_frames = embeddings.shape[0]
        
        # 2. Hitung Energy (RMS) Manual per Frame
        # YAMNet: Window=0.96s, Hop=0.48s. SR=16000.
        # Frame length = 16000 * 0.96 = 15360 samples
        # Hop length = 16000 * 0.48 = 7680 samples
        frame_length = 15360
        hop_length = 7680
        
        # Hitung RMS
        rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length, center=True)
        rms = rms[0] # (N_rms_frames,)
        
        # 3. Sinkronisasi Dimensi (PENTING!)
        # Kadang librosa RMS punya frame lebih banyak 1-2 frame dibanding YAMNet output
        # Kita potong agar sama panjang
        min_len = min(n_frames, len(rms))
        embeddings = embeddings[:min_len]
        rms = rms[:min_len]
        
        # Reshape RMS biar bisa dikalikan: (N, 1)
        rms = rms.reshape(-1, 1)
        
        # Hindari pembagian nol
        total_energy = np.sum(rms) + 1e-8
        
        # 4. RUMUS WEIGHTED POOLING
        # "Bagian keras lebih berpengaruh, bagian hening diabaikan"
        # Weighted Mean = Sum(Emb * Energy) / Sum(Energy)
        weighted_mean = np.sum(embeddings * rms, axis=0) / total_energy
        
        # 5. Tambahkan Max Pooling (Untuk menangkap puncak emosi)
        max_emb = np.max(embeddings, axis=0)
        
        # Gabung: [Weighted_Mean (1024) + Max (1024)] = 2048 Dimensi
        final_emb = np.concatenate([weighted_mean, max_emb], axis=0)
        
        return final_emb
        
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None

def load_subset(yamnet, subset_name):
    X, y, filenames = [], [], []
    target_dir = DATA_DIR / subset_name
    
    if not target_dir.exists(): return np.array(X), np.array(y), filenames

    moods = sorted(os.listdir(target_dir))
    for mood in moods:
        mood_dir = target_dir / mood
        if not mood_dir.is_dir(): continue
        files = sorted([f for f in os.listdir(mood_dir) if f.endswith('.wav')])
        
        for f in tqdm(files, leave=False, desc=f"Loading {mood}"):
            emb = extract_energy_features(yamnet, mood_dir / f)
            if emb is not None:
                X.append(emb)
                y.append(mood)
                filenames.append(f)
    return np.array(X), np.array(y), filenames

def plot_cm(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Energy-Aware Weighted Pooling')
    plt.xlabel('Prediksi'); plt.ylabel('Asli')
    plt.tight_layout()
    plt.savefig("cm_exp4_energy.png")
    plt.show()

def main():
    if not MLRUNS_DIR.exists(): MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Exp4_Energy_Aware")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        mlflow.log_param("dataset_path", str(DATA_DIR))
        mlflow.log_param("seed", 42)
        mlflow.log_param("feature_mode", "Energy Weighted Mean + Max (2048 dim)")
        
        yamnet = load_yamnet()
        print("📥 Loading Dataset (Energy-Weighted Features)...")
        X_train, y_train_txt, train_files = load_subset(yamnet, 'train')
        X_test, y_test_txt, test_files = load_subset(yamnet, 'test')
        
        le = LabelEncoder()
        y_train = to_categorical(le.fit_transform(y_train_txt))
        y_test = to_categorical(le.transform(y_test_txt))
        classes = le.classes_
        
        print(f"📊 Input Shape: {X_train.shape}") # Harusnya (N, 2048)

        # Class Weights (Wajib ada biar SAD tidak hilang)
        y_integers = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weights_dict = dict(enumerate(class_weights))
        print(f"⚖️ Class Weights: {class_weights_dict}")

        # --- MODEL DEFINITION ---
        model = Sequential([
            # Input 2048 (Weighted Mean + Max)
            Dense(1024, activation='relu', input_shape=(2048,)), 
            BatchNormalization(),
            Dropout(0.5), 
            
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(len(classes), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
        
        print("🚀 Mulai Training (Energy-Aware)...")
        model.fit(X_train, y_train, epochs=100, batch_size=16, 
                  validation_data=(X_test, y_test), 
                  callbacks=[early_stop],
                  class_weight=class_weights_dict)

        # Evaluasi
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n🏆 Final Accuracy: {acc*100:.2f}%")
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(y_true, y_pred, target_names=classes))
        
        plot_cm(y_true, y_pred, classes)
        mlflow.log_artifact("cm_exp4_energy.png")
        
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_DIR / 'model_exp4_energy.h5')
        np.save(MODEL_SAVE_DIR / 'classes.npy', classes)
        print(f"💾 Model tersimpan di: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()