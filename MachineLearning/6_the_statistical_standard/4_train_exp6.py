import os
import random
import numpy as np
import tensorflow as tf

# --- 1. KUNCI RANDOMNESS (SEED) ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"🔒 Random Seed dikunci di angka: {seed}")

set_seed(42) 

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
DATA_DIR = PROJECT_DIR / 'data' / 'processed_exp6' # Folder Baru Exp 6
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_exp6' # Simpan model terpisah
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000 

def load_yamnet(): 
    return hub.load(YAMNET_URL)

# --- 2. FITUR MEAN + STD (IDEAL MINIMAL) ---
def extract_features_mean_std(yamnet, file_path):
    try:
        wav, _ = librosa.load(file_path, sr=SR)
        if np.max(np.abs(wav)) > 0: 
            wav = wav / np.max(np.abs(wav))
        
        # Padding jika kependekan
        if len(wav) < 15600: 
            wav = np.pad(wav, (0, 15600 - len(wav)))
            
        _, embeddings, _ = yamnet(wav)
        
        # --- POOLING STRATEGY ---
        # 1. Mean (Dominan Mood)
        mean_emb = tf.reduce_mean(embeddings, axis=0)
        # 2. Std (Dinamika)
        std_emb = tf.math.reduce_std(embeddings, axis=0)
        
        # Gabung: 1024 + 1024 = 2048 Dimensi
        final_emb = tf.concat([mean_emb, std_emb], axis=0)
        
        return final_emb.numpy()
        
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
            emb = extract_features_mean_std(yamnet, mood_dir / f)
            if emb is not None:
                X.append(emb)
                y.append(mood)
                filenames.append(f)
    return np.array(X), np.array(y), filenames

def plot_cm(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Exp 6 (Mean + Std)')
    plt.xlabel('Prediksi'); plt.ylabel('Asli')
    plt.tight_layout()
    plt.savefig("cm_exp6.png")
    plt.show()

def main():
    if not MLRUNS_DIR.exists(): MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Exp6_Mean_Std_Standard")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        mlflow.log_param("dataset_path", str(DATA_DIR))
        mlflow.log_param("seed", 42)
        mlflow.log_param("pooling", "Mean + Std (2048 dim)")
        
        yamnet = load_yamnet()
        print("📥 Loading Dataset (Exp 6)...")
        X_train, y_train_txt, train_files = load_subset(yamnet, 'train')
        X_test, y_test_txt, test_files = load_subset(yamnet, 'test')
        
        le = LabelEncoder()
        y_train = to_categorical(le.fit_transform(y_train_txt))
        y_test = to_categorical(le.transform(y_test_txt))
        classes = le.classes_
        
        print(f"📊 Input Shape: {X_train.shape}") # Harusnya (N, 2048)

        # Class Weights (Wajib untuk dataset kecil/imbalance)
        y_integers = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weights_dict = dict(enumerate(class_weights))
        print(f"⚖️ Class Weights: {class_weights_dict}")

        # --- MODEL ARCHITECTURE ---
        model = Sequential([
            # Input Layer (2048 features)
            Dense(1024, activation='relu', input_shape=(2048,)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Hidden Layer
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            # Output Layer
            Dense(len(classes), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
        
        print("🚀 Mulai Training Exp 6...")
        model.fit(X_train, y_train, epochs=100, batch_size=16, 
                  validation_data=(X_test, y_test), 
                  callbacks=[early_stop],
                  class_weight=class_weights_dict)

        # Evaluasi
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n🏆 Final Accuracy Exp 6: {acc*100:.2f}%")
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(y_true, y_pred, target_names=classes))
        
        plot_cm(y_true, y_pred, classes)
        mlflow.log_artifact("cm_exp6.png")
        
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_DIR / 'model_exp6.h5')
        np.save(MODEL_SAVE_DIR / 'classes.npy', classes)
        print(f"💾 Model tersimpan di: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()