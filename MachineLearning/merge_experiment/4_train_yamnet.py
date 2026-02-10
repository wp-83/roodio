import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import mlflow
import mlflow.tensorflow
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]

# Folder Input
DATA_DIR = PROJECT_DIR / 'data' / 'interim_hybrid_merge'
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_yamnet'

# URL YAMNet
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000 

def load_yamnet():
    print("📥 Loading YAMNet from TF Hub...")
    return hub.load(YAMNET_URL)

def extract_features(yamnet, file_path):
    try:
        wav, _ = librosa.load(file_path, sr=SR)
        max_val = np.max(np.abs(wav))
        if max_val > 0: wav = wav / max_val
        else: return None
        
        if len(wav) < 15600: 
            wav = np.pad(wav, (0, 15600 - len(wav)))
        
        scores, embeddings, spectrogram = yamnet(wav)
        global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        return global_embedding
    except Exception as e:
        return None

def load_subset(yamnet, subset_name):
    X = []
    y = []
    target_dir = DATA_DIR / subset_name
    
    print(f"\n🎵 Loading {subset_name.upper()} Data...")
    if not target_dir.exists():
        print(f"❌ Error: Folder {target_dir} tidak ditemukan!")
        return np.array([]), np.array([])

    moods = sorted(os.listdir(target_dir))
    
    for mood in moods:
        mood_dir = target_dir / mood
        if not mood_dir.is_dir(): continue
        files = [f for f in os.listdir(mood_dir) if f.endswith('.wav')]
        
        for f in tqdm(files, leave=False):
            emb = extract_features(yamnet, mood_dir / f)
            if emb is not None:
                X.append(emb)
                y.append(mood)
                
    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Asli (Lokal)')
    plt.title('Confusion Matrix: Hybrid Model Final')
    plt.show()

def main():
    if not MLRUNS_DIR.exists(): MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Roodio_Hybrid_Tuned_Merge")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        yamnet = load_yamnet()
        
        X_train, y_train_text = load_subset(yamnet, 'train')
        X_test, y_test_text = load_subset(yamnet, 'test')
        
        if len(X_train) == 0: return

        le = LabelEncoder()
        y_train = to_categorical(le.fit_transform(y_train_text))
        y_test = to_categorical(le.transform(y_test_text))
        classes = le.classes_
        
        # --- ARSITEKTUR TUNING (V2) ---
        model = Sequential([
            # Layer 1: Lebih simple tapi pakai Regularizer L2
            Dense(512, activation='relu', input_shape=(1024,), 
                  kernel_regularizer=regularizers.l2(0.001)), # Mencegah bobot terlalu besar
            BatchNormalization(), # Menstabilkan training
            Dropout(0.4),         # Dropout moderat
            
            # Layer 2
            Dense(256, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output Layer
            Dense(len(classes), activation='softmax')
        ])
        
        # Optimizer dengan Learning Rate yang fleksibel
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Loss dengan Label Smoothing (PENTING untuk data emosi)
        # label_smoothing=0.1 artinya label 1.0 dianggap 0.9.
        # Ini mencegah model terlalu 'sombong' dan membantu generalisasi Relaxed/Sad.
        model.compile(optimizer=optimizer, 
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
                      metrics=['accuracy'])
        
        # --- CALLBACKS PRO ---
        # 1. ReduceLROnPlateau: Kalau akurasi stuck 5 epoch, turunkan LR jadi 20%-nya
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        
        # 2. EarlyStopping: Kalau stuck 15 epoch, stop.
        early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
        
        print("\n🚀 Mulai Training (Tuned Version)...")
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr] # Tambahkan ReduceLR
        )
        
        # Evaluasi
        print("\n" + "="*50)
        print("📝 HASIL EVALUASI HYBRID TUNED")
        print("="*50)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"🏆 Final Accuracy (Local Data): {acc*100:.2f}%")
        
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=classes))
        
        plot_confusion_matrix(y_true, y_pred, classes)
        
        # Simpan
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_DIR / 'mood_classifier.h5') # Format H5 standar
        np.save(MODEL_SAVE_DIR / 'classes.npy', classes)
        print(f"\n💾 Model tersimpan di: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()