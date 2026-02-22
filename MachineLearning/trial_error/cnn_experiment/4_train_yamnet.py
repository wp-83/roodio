import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

# PENTING: Arahkan ke folder SEGMEN yang baru dibuat
DATA_DIR = PROJECT_DIR / 'data' / 'processed_yamnet_segments'
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_yamnet'

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000

def load_yamnet():
    print("📥 Loading YAMNet...")
    return hub.load(YAMNET_URL)

def extract_embedding(yamnet, file_path):
    """
    Input: File WAV 3 Detik (16kHz)
    Output: Embedding Vector (1024,)
    """
    try:
        # Load (Cepat, karena sudah 16k dan pendek)
        wav, _ = librosa.load(file_path, sr=SR)
        
        # Normalisasi
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val
        else:
            return None
            
        # Padding jika < 0.975 detik (syarat minimal YAMNet)
        if len(wav) < 15600: 
            wav = np.pad(wav, (0, 15600 - len(wav)))

        # YAMNet Inference
        scores, embeddings, spectrogram = yamnet(wav)
        
        # Karena inputnya pendek (3 detik), output embeddings mungkin ada 3-4 baris.
        # Kita rata-rata biar jadi 1 vektor.
        global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        return global_embedding
    except Exception as e:
        # print(f"Err: {e}") # Silent error biar gak spam console
        return None

def load_dataset(yamnet, subset):
    X = []
    y = []
    target_dir = DATA_DIR / subset
    
    print(f"\n🎵 Loading {subset.upper()} Segments...")
    moods = sorted(os.listdir(target_dir))
    
    for mood in moods:
        mood_path = target_dir / mood
        if not mood_path.is_dir(): continue
        
        files = [f for f in os.listdir(mood_path) if f.endswith('.wav')]
        # Limit log biar gak penuh (opsional)
        print(f"   📂 {mood}: {len(files)} segments")
        
        for f in tqdm(files, leave=False):
            emb = extract_embedding(yamnet, mood_path / f)
            if emb is not None:
                X.append(emb)
                y.append(mood)
                
    return np.array(X), np.array(y)

def main():
    if not MLRUNS_DIR.exists(): MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Roodio_YAMNet_Segmented")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        yamnet = load_yamnet()
        
        # 1. Load Data (Sekarang jumlahnya ribuan segmen!)
        X_train, y_train_text = load_dataset(yamnet, 'train')
        X_test, y_test_text = load_dataset(yamnet, 'test')
        
        print(f"\n📊 Data Stats:")
        print(f"   Train Segments: {X_train.shape[0]}")
        print(f"   Test Segments : {X_test.shape[0]}")
        
        # 2. Encode
        le = LabelEncoder()
        y_train = to_categorical(le.fit_transform(y_train_text))
        y_test = to_categorical(le.transform(y_test_text))
        classes = le.classes_
        
        # 3. Model Classifier
        model = Sequential([
            Dense(512, activation='relu', input_shape=(1024,)),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(len(classes), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # 4. Training
        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        
        print("\n🚀 Training on Segments...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64, # Batch lebih besar karena data banyak
            validation_data=(X_test, y_test),
            callbacks=[early_stop]
        )
        
        # 5. Evaluasi
        loss, acc = model.evaluate(X_test, y_test)
        print(f"\n🏆 SEGMENT ACCURACY: {acc*100:.2f}%")
        
        # Classification Report
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        print(classification_report(y_true, y_pred, target_names=classes))
        
        # Simpan
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_DIR / 'mood_classifier.h5')
        np.save(MODEL_SAVE_DIR / 'classes.npy', classes)
        print("💾 Model Saved!")

if __name__ == "__main__":
    main()