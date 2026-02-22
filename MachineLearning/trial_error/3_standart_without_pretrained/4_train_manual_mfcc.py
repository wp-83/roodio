import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]
DATA_DIR = PROJECT_DIR / 'data' / 'processed_manual' # Data tanpa augmentasi
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_manual_mfcc'

# Setting Manual Features
SR = 22050 
N_MFCC = 40 # Kita ambil 40 fitur akustik dasar

def extract_manual_features(file_path):
    """Pengganti YAMNet: Hitung MFCC Manual"""
    try:
        y, sr = librosa.load(file_path, sr=SR)
        # 1. Hitung MFCC (Mel-frequency cepstral coefficients)
        # Ini merepresentasikan 'timbre' atau warna suara
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        
        # 2. Ambil Rata-rata (Mean Pooling) biar dimensinya 1D
        # Dari (40, time) menjadi (40,)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        return mfcc_mean
    except: return None

def load_subset(subset_name):
    X, y, filenames = [], [], []
    target_dir = DATA_DIR / subset_name
    moods = sorted(os.listdir(target_dir))
    
    for mood in moods:
        mood_dir = target_dir / mood
        if not mood_dir.is_dir(): continue
        files = [f for f in os.listdir(mood_dir) if f.endswith('.wav')]
        
        for f in tqdm(files, leave=False, desc=f"Extracting MFCC {mood}"):
            feat = extract_manual_features(mood_dir / f)
            if feat is not None:
                X.append(feat)
                y.append(mood)
                filenames.append(f)
    return np.array(X), np.array(y), filenames

def plot_cm(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Manual MFCC (No YAMNet)')
    plt.xlabel('Prediksi'); plt.ylabel('Asli')
    plt.savefig("cm_manual.png")
    plt.show()

def main():
    if not MLRUNS_DIR.exists(): MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Baseline_Manual_MFCC")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        # 1. Log Parameter (LENGKAP)
        mlflow.log_param("dataset_path", str(DATA_DIR))
        mlflow.log_param("data_split", "72 Train / 28 Test")
        mlflow.log_param("augmentation", "FALSE")
        mlflow.log_param("feature_extractor", f"Manual MFCC (n={N_MFCC})")
        mlflow.log_param("transfer_learning", "FALSE (No YAMNet)")
        
        print("📥 Extracting Features (MFCC)...")
        X_train, y_train_txt, train_files = load_subset('train')
        X_test, y_test_txt, test_files = load_subset('test')
        
        # 2. Upload Daftar File
        with open("manifest_mfcc.txt", "w") as f:
            f.write(f"Total Train: {len(X_train)} | Total Test: {len(X_test)}\n\n")
            for i, fname in enumerate(test_files):
                f.write(f"TEST: {y_test_txt[i]} - {fname}\n")
        mlflow.log_artifact("manifest_mfcc.txt")

        le = LabelEncoder()
        y_train = to_categorical(le.fit_transform(y_train_txt))
        y_test = to_categorical(le.transform(y_test_txt))
        classes = le.classes_

        # 3. Model Arsitektur (Disesuaikan untuk Input Kecil)
        # Input shape sekarang (40,) bukan (1024,)
        model = Sequential([
            Input(shape=(N_MFCC,)), 
            
            # Layer 1
            Dense(256, activation='relu'), # Ukuran dikecilkan (512 -> 256) krn input kecil
            BatchNormalization(),
            Dropout(0.4),
            
            # Layer 2
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output
            Dense(len(classes), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
        
        print(f"🚀 Mulai Training Manual (Input Features: {N_MFCC})...")
        model.fit(X_train, y_train, epochs=100, batch_size=16, 
                  validation_data=(X_test, y_test), callbacks=[early_stop])

        # 4. Evaluasi
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n🏆 Manual MFCC Accuracy: {acc*100:.2f}%")
        mlflow.log_metric("final_test_accuracy", acc)
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print(classification_report(y_true, y_pred, target_names=classes))
        plot_cm(y_true, y_pred, classes)
        mlflow.log_artifact("cm_manual.png")
        
        # Simpan
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_DIR / 'model_mfcc.h5')
        np.save(MODEL_SAVE_DIR / 'classes.npy', classes)

if __name__ == "__main__":
    main()