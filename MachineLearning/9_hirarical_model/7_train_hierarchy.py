import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Import modul fitur kita
import features_hierarchy as feat

# --- UPDATE PATH DI SINI ---
# Kita arahkan ke folder TRAIN hasil prepare tadi
BASE_DIR = 'data/hierarchy/train' 
MODELS_DIR = 'models_hierarchy'
os.makedirs(MODELS_DIR, exist_ok=True)

def train_stage(stage_name, extract_func, hidden_units=[512, 256]):
    print(f"\n🚀 TRAINING {stage_name.upper()}...")
    data_path = os.path.join(BASE_DIR, stage_name)
    
    # Cek folder sebelum lanjut biar gak error kayak tadi
    if not os.path.exists(data_path):
        print(f"❌ Error: Folder tidak ditemukan: {data_path}")
        print("   Pastikan sudah menjalankan '6_prepare_hierarchy.py'!")
        return

    X, y = [], []
    classes = sorted(os.listdir(data_path))
    
    # 1. Load Data
    for label in classes:
        folder = os.path.join(data_path, label)
        files = os.listdir(folder)
        for f in tqdm(files, desc=f"Loading {label}"):
            if not f.endswith('.wav'): continue
            try:
                vector = extract_func(os.path.join(folder, f))
                if vector is not None:
                    X.append(vector)
                    y.append(label)
            except Exception as e:
                print(f"Error {f}: {e}")
                
    X = np.array(X)
    le = LabelEncoder()
    y_enc = to_categorical(le.fit_transform(y))
    
    # 2. SHUFFLE DATA (WAJIB untuk Validation Split)
    # Agar data teracak sebelum dipotong 20% buat validasi
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y_enc = y_enc[indices]
    
    print(f"📊 Input Shape: {X.shape} | Classes: {le.classes_}")
    
    # 3. Class Weights
    y_int = np.argmax(y_enc, axis=1)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
    weights_dict = dict(enumerate(weights))
    
    # 4. Model Binary
    model = Sequential([
        Dense(hidden_units[0], activation='relu', input_shape=(X.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(hidden_units[1], activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(2, activation='softmax') # Binary Output
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 5. Early Stopping (Pantau Val Loss)
    early_stop = EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', verbose=1)
    
    # 6. Fit dengan Validation Split (20%)
    history = model.fit(
        X, y_enc, 
        epochs=100, 
        batch_size=16, 
        validation_split=0.2, # Ambil 20% otomatis buat validasi
        callbacks=[early_stop], 
        class_weight=weights_dict, 
        verbose=1
    )
    
    # Save Model
    model.save(os.path.join(MODELS_DIR, f'model_{stage_name}.h5'))
    np.save(os.path.join(MODELS_DIR, f'classes_{stage_name}.npy'), le.classes_)
    
    # Print Info Akhir
    if len(history.history['val_accuracy']) > 0:
        final_acc = history.history['val_accuracy'][-1] * 100
        print(f"💾 Model {stage_name} Saved! (Val Acc: {final_acc:.2f}%)")
    else:
        print(f"💾 Model {stage_name} Saved!")

if __name__ == "__main__":
    # Train 3 Model Berurutan
    # 1. Arousal (High vs Low)
    train_stage('stage_1_arousal', feat.extract_stage_1, hidden_units=[1024, 512])
    
    # 2. High Arousal Group (Angry vs Happy)
    train_stage('stage_2a_high_group', feat.extract_stage_2a)
    
    # 3. Low Arousal Group (Sad vs Relaxed)
    train_stage('stage_2b_low_group', feat.extract_stage_2b)