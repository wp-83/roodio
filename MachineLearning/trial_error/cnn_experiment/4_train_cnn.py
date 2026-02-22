import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# --- KONFIGURASI PATH ---
# Menggunakan pathlib agar path dinamis (aman dijalankan dari root atau subfolder)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1] # Naik 1 level ke root project
DATA_DIR = PROJECT_DIR / 'data' / 'processed_cnn'
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_cnn'

# Konfigurasi Training
IMG_HEIGHT = 128  # Sesuai n_mels
IMG_WIDTH = 130   # Kira-kira 3 detik audio @ 22050hz hop 512
BATCH_SIZE = 32
EPOCHS = 20

def setup_mlflow():
    """Konfigurasi MLflow agar log masuk ke folder mlruns utama"""
    if not MLRUNS_DIR.exists():
        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Roodio_Mood_CNN_Spectrogram")
    
    # AUTOLOG: Otomatis catat grafik accuracy, loss, dan model
    mlflow.tensorflow.autolog(log_models=True)
    print(f"✅ MLflow Tracking URI: {mlflow.get_tracking_uri()}")

def load_dataset(subset):
    X = []
    y = []
    base_path = DATA_DIR / subset
    
    if not base_path.exists():
        raise FileNotFoundError(f"Folder data tidak ditemukan di: {base_path}")

    moods = sorted(os.listdir(base_path)) 
    print(f"⏳ Loading {subset} data (Moods: {moods})...")
    
    for mood in moods:
        mood_path = base_path / mood
        if not mood_path.is_dir(): continue

        files = [f for f in os.listdir(mood_path) if f.endswith('.npy')]
        
        for f in files:
            # Load gambar spektrogram
            spectrogram = np.load(mood_path / f)
            
            # Padding / Trimming biar ukuran konsisten 128x130
            if spectrogram.shape[1] < IMG_WIDTH:
                pad_width = IMG_WIDTH - spectrogram.shape[1]
                spectrogram = np.pad(spectrogram, ((0,0), (0, pad_width)))
            else:
                spectrogram = spectrogram[:, :IMG_WIDTH]
                
            X.append(spectrogram)
            y.append(mood)
            
    # Ubah ke format Array 4 Dimensi (Batch, Height, Width, Channel)
    X = np.array(X)[..., np.newaxis] 
    return X, np.array(y)

def main():
    setup_mlflow()
    
    # Mulai Run MLflow
    with mlflow.start_run(run_name="CNN_MelSpectrogram_Baseline"):
        
        # 1. Load Data
        print(f"📂 Mengambil data dari: {DATA_DIR}")
        try:
            X_train, y_train_text = load_dataset('train')
            X_test, y_test_text = load_dataset('test')
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("   Jalankan '3_slice_segments.py' dulu!")
            return

        print(f"📊 Data Train Shape: {X_train.shape}")
        print(f"📊 Data Test Shape : {X_test.shape}")
        
        # 2. Encode Label
        le = LabelEncoder()
        y_train = to_categorical(le.fit_transform(y_train_text))
        y_test  = to_categorical(le.transform(y_test_text))
        
        print("✅ Classes:", le.classes_)

        # 3. Buat Model CNN
        model = Sequential([
            # Layer 1
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
            MaxPooling2D((2, 2)),
            
            # Layer 2
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Layer 3
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3), 
            Dense(4, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # 4. Training
        print("🚀 Mulai Training CNN...")
        # MLflow Autolog akan otomatis menangkap history training di sini
        history = model.fit(X_train, y_train, epochs=EPOCHS, 
                            validation_data=(X_test, y_test), 
                            batch_size=BATCH_SIZE)
        
        # 5. Evaluasi Manual (Untuk Log Metric Final)
        print("\n📝 Evaluasi Akhir:")
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"🏆 Akurasi Segmen (3 detik): {test_acc*100:.2f}%")
        
        # Log metric manual ke MLflow (biar mudah dicari di tabel)
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_loss", test_loss)
        
        # Prediksi Detail & Classification Report
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        report = classification_report(y_true, y_pred, target_names=le.classes_)
        print(report)
        
        # Simpan Report ke text file lalu upload ke MLflow Artifacts
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        # 6. Simpan Model Lokal (Backup)
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        model_path = MODEL_SAVE_DIR / 'audio_mood_cnn.h5'
        classes_path = MODEL_SAVE_DIR / 'classes.npy'
        
        model.save(model_path)
        np.save(classes_path, le.classes_)
        
        print(f"💾 Model tersimpan lokal di: {model_path}")

if __name__ == "__main__":
    main()