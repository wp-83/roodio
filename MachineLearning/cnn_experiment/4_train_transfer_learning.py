import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURASI PATH ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]
DATA_DIR = PROJECT_DIR / 'data' / 'processed_cnn'
MLRUNS_DIR = PROJECT_DIR / 'mlruns'
MODEL_SAVE_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_cnn'

# Konfigurasi Training
IMG_HEIGHT = 128
IMG_WIDTH = 130
BATCH_SIZE = 32

def setup_mlflow():
    if not MLRUNS_DIR.exists(): MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("Roodio_Mood_Transfer_2Stage")
    mlflow.tensorflow.autolog(log_models=True)

def load_dataset(subset):
    X = []
    y = []
    base_path = DATA_DIR / subset
    if not base_path.exists(): raise FileNotFoundError(f"Folder {base_path} hilang!")

    moods = sorted(os.listdir(base_path))
    print(f"⏳ Loading {subset} data...")
    
    for mood in moods:
        mood_path = base_path / mood
        if not mood_path.is_dir(): continue
        files = [f for f in os.listdir(mood_path) if f.endswith('.npy')]
        for f in files:
            spec = np.load(mood_path / f)
            if spec.shape[1] < IMG_WIDTH:
                pad_width = IMG_WIDTH - spec.shape[1]
                spec = np.pad(spec, ((0,0), (0, pad_width)))
            else:
                spec = spec[:, :IMG_WIDTH]
            X.append(spec)
            y.append(mood)
            
    X = np.array(X)[..., np.newaxis] 
    return X, np.array(y)

def main():
    setup_mlflow()
    
    with mlflow.start_run(run_name="MobileNetV2_2Stage_Fix"):
        # 1. Load Data
        X_train, y_train_text = load_dataset('train')
        X_test, y_test_text = load_dataset('test')
        
        le = LabelEncoder()
        y_train = to_categorical(le.fit_transform(y_train_text))
        y_test  = to_categorical(le.transform(y_test_text))
        print("✅ Classes:", le.classes_)

        # 2. DEFINISI MODEL DASAR
        input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
        # Adaptasi Channel (1 -> 3)
        x = Conv2D(3, (3, 3), padding='same', activation='relu')(input_tensor)
        
        # Load MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        # --- PHASE 1: BEKUKAN TOTAL (WARM UP) ---
        base_model.trainable = False 
        
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_tensor = Dense(4, activation='softmax')(x)
        
        model = Model(inputs=input_tensor, outputs=output_tensor)
        
        # Compile untuk Phase 1
        model.compile(optimizer=Adam(learning_rate=1e-3), # LR Standar
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        print("\n🔥 PHASE 1: WARM-UP (Melatih Head Saja)...")
        history1 = model.fit(X_train, y_train, epochs=10, 
                             validation_data=(X_test, y_test), 
                             batch_size=BATCH_SIZE)
        
        # --- PHASE 2: FINE TUNING (POLESAN AKHIR) ---
        print("\n🔓 PHASE 2: FINE-TUNING (Mencairkan Otak)...")
        
        base_model.trainable = True
        # Bekukan 100 layer pertama, cairkan sisanya
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Compile ulang dengan LR SANGAT KECIL
        model.compile(optimizer=Adam(learning_rate=1e-5), # LR Kecil
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        history2 = model.fit(X_train, y_train, epochs=20, # Lanjut training
                             validation_data=(X_test, y_test), 
                             batch_size=BATCH_SIZE,
                             callbacks=callbacks)
        
        # Simpan Model
        if not MODEL_SAVE_DIR.exists(): MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_SAVE_DIR / 'audio_mood_cnn.h5')
        np.save(MODEL_SAVE_DIR / 'classes.npy', le.classes_)
        print("💾 Model Final Tersimpan! Silakan jalankan '5_evaluate_voting.py'.")

if __name__ == "__main__":
    main()