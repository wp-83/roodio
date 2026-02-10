import os
import random # Wajib import
import numpy as np
import tensorflow as tf

# --- BAGIAN PENGUNCI RANDOM (SEED) - WAJIB PALING ATAS ---
def set_seed(seed=34):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"🔒 Random Seed dikunci: {seed}")

set_seed(42) # Eksekusi sebelum import library lain yang berat

import tensorflow_hub as hub
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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
DATA_DIR = PROJECT_DIR / 'data' / 'processed_basic' 
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_basic_fixed' # Ganti nama folder biar gak numpuk
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000 

def load_yamnet(): return hub.load(YAMNET_URL)

def extract_features(yamnet, file_path):
    try:
        wav, _ = librosa.load(file_path, sr=SR)
        if np.max(np.abs(wav)) > 0: wav = wav / np.max(np.abs(wav))
        if len(wav) < 15600: wav = np.pad(wav, (0, 15600 - len(wav)))
        _, embeddings, _ = yamnet(wav)
        return tf.reduce_mean(embeddings, axis=0).numpy()
    except: return None

def load_subset(yamnet, subset_name):
    X, y, filenames = [], [], []
    target_dir = DATA_DIR / subset_name
    
    # Sortir folder mood agar urutan pembacaan selalu sama
    moods = sorted(os.listdir(target_dir))
    
    for mood in moods:
        mood_dir = target_dir / mood
        if not mood_dir.is_dir(): continue
        # Sortir file agar urutan selalu sama
        files = sorted([f for f in os.listdir(mood_dir) if f.endswith('.wav')])
        
        for f in tqdm(files, leave=False, desc=f"Loading {mood}"):
            emb = extract_features(yamnet, mood_dir / f)
            if emb is not None:
                X.append(emb)
                y.append(mood)
                filenames.append(f)
    return np.array(X), np.array(y), filenames

def plot_cm(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Basic Local (Fixed Seed)')
    plt.xlabel('Prediksi'); plt.ylabel('Asli')
    plt.savefig("confusion_matrix_fixed.png")
    plt.show()

def main():
    if not MLRUNS_DIR.exists(): MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Basic_Local_Fixed")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        mlflow.log_param("dataset_path", str(DATA_DIR))
        mlflow.log_param("seed", 42) # Catat seed di MLflow
        
        yamnet = load_yamnet()
        print("📥 Loading Dataset...")
        X_train, y_train_txt, train_files = load_subset(yamnet, 'train')
        X_test, y_test_txt, test_files = load_subset(yamnet, 'test')
        
        le = LabelEncoder()
        y_train = to_categorical(le.fit_transform(y_train_txt))
        y_test = to_categorical(le.transform(y_test_txt))
        classes = le.classes_

        # Model Deep Learning Standar
        # Karena Seed dikunci, inisialisasi bobot awal (weight init) akan selalu sama
        model = Sequential([
            Dense(512, activation='relu', input_shape=(1024,)),
            BatchNormalization(),
            Dropout(0.5), # Pattern dropout akan sama
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(len(classes), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
        
        print("🚀 Mulai Training Basic (Fixed Seed)...")
        # Shuffle=True tetap bagus untuk training, tapi karena seed dikunci,
        # urutan pengacakannya akan selalu sama persis tiap kali run.
        model.fit(X_train, y_train, epochs=100, batch_size=16, 
                  validation_data=(X_test, y_test), callbacks=[early_stop])

        # Evaluasi
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n🏆 Basic Model Accuracy: {acc*100:.2f}%")
        mlflow.log_metric("final_test_accuracy", acc)
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print(classification_report(y_true, y_pred, target_names=classes))
        plot_cm(y_true, y_pred, classes)
        mlflow.log_artifact("confusion_matrix_fixed.png")
        
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_DIR / 'basic_model_fixed.h5')
        np.save(MODEL_SAVE_DIR / 'classes.npy', classes)
        print(f"💾 Model tersimpan di: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()