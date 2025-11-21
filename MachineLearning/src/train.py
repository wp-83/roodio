import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import src.config as cfg

def load_data():
    print("1. Memuat data dari file .npy...")
    # Load data yang sudah diproses (Cepat!)
    X = np.load(os.path.join(cfg.PROCESSED_DATA_DIR, "X_train.npy"))
    y = np.load(os.path.join(cfg.PROCESSED_DATA_DIR, "y_train.npy"))
    
    print(f"   Data dimuat: {X.shape} sampel")
    return X, y

def build_model(input_shape):
    print("2. Membangun Arsitektur CNN...")
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        # Flatten & Dense
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4), # Mencegah overfitting
        
        # Output Layer (2 Neuron: Valence & Arousal)
        # Gunakan linear/tanh tergantung range data. 
        # Dari EDA Anda, range data -0.8 s.d 0.8, jadi 'tanh' atau 'linear' cocok.
        # Kita pakai 'linear' agar fleksibel.
        Dense(2, activation='linear', name='valence_arousal_output')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001), # Learning rate kecil agar stabil
        loss='mse',                           # Mean Squared Error untuk Regresi
        metrics=['mae']                       # Mean Absolute Error untuk pantau akurasi
    )
    
    return model

def plot_history(history):
    """Fungsi untuk menggambar grafik training"""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Simpan grafik
    plot_path = os.path.join(cfg.BASE_DIR, "training_history.png")
    plt.savefig(plot_path)
    print(f"   Grafik training disimpan di: {plot_path}")

def main():
    # A. Load Data
    X, y = load_data()
    
    # B. Split Train/Test (80% Latih, 20% Ujian)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training Set: {X_train.shape[0]} | Validation Set: {X_val.shape[0]}")
    
    # C. Buat Model
    model = build_model(X_train.shape[1:]) # Input shape (128, 128, 1)
    model.summary()
    
    # D. Persiapan Training
    # Simpan model terbaik saja (jika val_loss turun)
    models_dir = os.path.join(cfg.BASE_DIR, "models")
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, "model_emosi_cnn.keras")
    
    checkpoint = ModelCheckpoint(
        model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    
    # Berhenti jika tidak ada perbaikan selama 5 epoch (biar gak buang waktu)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # E. MULAI TRAINING!
    print("\n3. Mulai Training (Bismillah)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,          # Maksimal 30 ronde
        batch_size=32,      # Sekali belajar 32 lagu
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # F. Selesai
    print("\n4. Training Selesai!")
    plot_history(history)
    print(f"✅ Model terbaik tersimpan di: {model_path}")

if __name__ == "__main__":
    main()