import os
import random
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

# --- 1. KUNCI RANDOMNESS ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(42)

import tensorflow_hub as hub
import librosa
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow

# --- KONFIGURASI ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]
DATA_DIR = PROJECT_DIR / 'data' / 'processed_exp4' 
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_exp4_voting'
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000 

def load_yamnet(): return hub.load(YAMNET_URL)

# --- 2. PERUBAHAN EKSTRAKSI FITUR (NO POOLING) ---
def extract_frames(yamnet, file_path):
    """Mengembalikan SEMUA frame (N, 1024), tidak dirata-rata"""
    try:
        wav, _ = librosa.load(file_path, sr=SR)
        if np.max(np.abs(wav)) > 0: wav = wav / np.max(np.abs(wav))
        
        # Min length check
        if len(wav) < 15600: wav = np.pad(wav, (0, 15600 - len(wav)))
            
        _, embeddings, _ = yamnet(wav)
        return embeddings.numpy() # Return (Jumlah_Frame, 1024)
    except: return None

# --- 3. DATA LOADER KHUSUS ---
def load_data_frames(yamnet, subset_name):
    """
    Untuk TRAINING: Kita butuh semua frame ditumpuk jadi satu.
    Untuk TESTING: Kita butuh list of arrays (per lagu) untuk voting.
    """
    X_flat = [] # Untuk training (semua frame digabung)
    y_flat = [] 
    
    X_grouped = [] # Untuk testing (dikelompokkan per lagu)
    y_grouped = []
    
    target_dir = DATA_DIR / subset_name
    moods = sorted(os.listdir(target_dir))
    
    for mood in moods:
        mood_dir = target_dir / mood
        if not mood_dir.is_dir(): continue
        files = sorted([f for f in os.listdir(mood_dir) if f.endswith('.wav')])
        
        for f in tqdm(files, leave=False, desc=f"Loading {mood}"):
            frames = extract_frames(yamnet, mood_dir / f)
            if frames is not None:
                # Simpan untuk Training (Flatten)
                for frame in frames:
                    X_flat.append(frame)
                    y_flat.append(mood)
                
                # Simpan untuk Voting (Grouped)
                X_grouped.append(frames)
                y_grouped.append(mood)
                
    return np.array(X_flat), np.array(y_flat), X_grouped, y_grouped, moods

def plot_cm(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Segment Voting')
    plt.xlabel('Prediksi'); plt.ylabel('Asli')
    plt.savefig("cm_exp4_voting.png")
    plt.show()

def main():
    if not MLRUNS_DIR.exists(): MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Exp4_Segment_Voting")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        yamnet = load_yamnet()
        print("📥 Loading Dataset (Frame-by-Frame)...")
        
        # Load Data
        X_train_flat, y_train_flat, _, _, _ = load_data_frames(yamnet, 'train')
        # Kita butuh grouped data untuk test set agar bisa voting
        _, _, X_test_grouped, y_test_grouped, mood_list = load_data_frames(yamnet, 'test')
        
        # Label Encoding
        le = LabelEncoder()
        le.fit(mood_list) # Fit dengan semua mood yg ada
        
        y_train_enc = to_categorical(le.transform(y_train_flat))
        classes = le.classes_
        
        print(f"📊 Training Frames: {X_train_flat.shape[0]} frames")
        
        # Class Weights (Frame level)
        y_int = np.argmax(y_train_enc, axis=1)
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
        weights_dict = dict(enumerate(weights))
        print(f"⚖️ Frame Weights: {weights_dict}")

        # --- MODEL SEDERHANA (Per Frame) ---
        # Model ini menilai mood per 0.96 detik. Jadi tidak perlu layer besar.
        model = Sequential([
            Dense(256, activation='relu', input_shape=(1024,)), # Input raw YAMNet
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(len(classes), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("🚀 Training Frame Classifier...")
        # Train pada ribuan frame sekaligus
        model.fit(X_train_flat, y_train_enc, 
                  epochs=50, 
                  batch_size=64, # Batch size lebih besar karena data banyak
                  class_weight=weights_dict,
                  verbose=1)

        # --- EVALUASI DENGAN VOTING ---
        print("\n🗳️ Melakukan Voting per Lagu...")
        y_pred_final = []
        y_true_final = []
        
        # Loop per lagu di test set
        for i, frames in enumerate(X_test_grouped):
            # 1. Prediksi setiap frame di lagu ini
            preds = model.predict(frames, verbose=0) 
            pred_classes = np.argmax(preds, axis=1)
            
            # 2. Lakukan Voting (Majority Vote)
            # Contoh: [0, 0, 1, 0, 2] -> Pemenang 0
            counts = Counter(pred_classes)
            winner = counts.most_common(1)[0][0]
            
            y_pred_final.append(winner)
            
            # Label asli lagu ini
            true_label_str = y_test_grouped[i]
            y_true_final.append(le.transform([true_label_str])[0])

        # Hitung Akurasi Lagu (Bukan Akurasi Frame)
        y_pred_final = np.array(y_pred_final)
        y_true_final = np.array(y_true_final)
        
        acc = np.mean(y_pred_final == y_true_final)
        print(f"\n🏆 Voting Accuracy: {acc*100:.2f}%")
        
        print("\n--- CLASSIFICATION REPORT (Song Level) ---")
        print(classification_report(y_true_final, y_pred_final, target_names=classes))
        
        plot_cm(y_true_final, y_pred_final, classes)
        mlflow.log_artifact("cm_exp4_voting.png")
        
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_DIR / 'model_exp4_voting.h5')
        np.save(MODEL_SAVE_DIR / 'classes.npy', classes)

if __name__ == "__main__":
    main()