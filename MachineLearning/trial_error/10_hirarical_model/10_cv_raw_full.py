import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import extractor kita
import features_hierarchy as feat

# --- KONFIGURASI ---
# GANTI INI sesuai lokasi folder RAW DATA LAMA anda
RAW_DATA_DIR = 'data/raw' 
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI EKSPERIMEN 10 (RAW DATA + {FOLDS}-FOLD CV)...")

# --- 1. PRE-CALCULATE FEATURES ---
# Kita ekstrak fitur sekali saja biar K-Fold nya cepat
print("📥 Mengekstrak fitur dari Raw Data (ini akan memakan waktu)...")

X_paths = []    # Simpan path file
y_labels = []   # Simpan label asli
cache_s1 = {}   # Cache fitur Stage 1
cache_s2a = {}  # Cache fitur Stage 2A
cache_s2b = {}  # Cache fitur Stage 2B

moods = ['angry', 'happy', 'sad', 'relaxed']

for mood in moods:
    folder = os.path.join(RAW_DATA_DIR, mood)
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Folder {folder} tidak ditemukan!")
        continue
        
    files = os.listdir(folder)
    for f in tqdm(files, desc=f"Extracting {mood.upper()}"):
        if not f.endswith(('.wav', '.mp3')): continue
        
        file_path = os.path.join(folder, f)
        
        # Ekstrak 3 Jenis Fitur sekaligus & Simpan di Cache
        # Biar nanti pas training loop gak perlu ekstrak ulang
        vec1 = feat.extract_stage_1(file_path)
        if vec1 is None: continue # Skip file rusak
        
        vec2a = feat.extract_stage_2a(file_path)
        vec2b = feat.extract_stage_2b(file_path)
        
        # Simpan
        X_paths.append(file_path) # Kita pakai path sebagai ID
        y_labels.append(mood)
        
        cache_s1[file_path] = vec1
        cache_s2a[file_path] = vec2a
        cache_s2b[file_path] = vec2b

X_paths = np.array(X_paths)
y_labels = np.array(y_labels)

print(f"✅ Total Data Valid: {len(X_paths)} files.")

# --- 2. HELPER: CREATE MODEL ---
def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 3. TRAINING FUNCTION PER STAGE ---
def train_stage(train_indices, stage_type, cache_data):
    """
    Melatih model untuk satu stage menggunakan data training index.
    stage_type: 'arousal', 'high', 'low'
    """
    X_train = []
    y_train = []
    
    # Filter Data Training
    for i in train_indices:
        path = X_paths[i]
        label = y_labels[i]
        
        if stage_type == 'arousal':
            # Semua data masuk, label jadi high/low
            target = 'high' if label in ['angry', 'happy'] else 'low'
            X_train.append(cache_data[path])
            y_train.append(target)
            
        elif stage_type == 'high':
            # Cuma Angry/Happy
            if label in ['angry', 'happy']:
                X_train.append(cache_data[path])
                y_train.append(label)
                
        elif stage_type == 'low':
            # Cuma Sad/Relaxed
            if label in ['sad', 'relaxed']:
                X_train.append(cache_data[path])
                y_train.append(label)
    
    X_train = np.array(X_train)
    if len(X_train) == 0: return None, None # Safety check
    
    # Encode & Weighting
    le = LabelEncoder()
    y_enc = to_categorical(le.fit_transform(y_train))
    
    y_int = np.argmax(y_enc, axis=1)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
    weights_dict = dict(enumerate(weights))
    
    # Train
    model = create_model(X_train.shape[1], 2)
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_enc, epochs=50, batch_size=16, 
              callbacks=[early_stop], class_weight=weights_dict, verbose=0)
    
    return model, le

# --- 4. START 5-FOLD LOOP ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

fold_accuracies = []
confusion_matrices = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_paths, y_labels)):
    print(f"\n🔄 --- FOLD {fold+1} / {FOLDS} ---")
    
    # A. TRAIN 3 MODEL
    print("   🏋️ Training Stage 1 (Arousal)...")
    m1, le1 = train_stage(train_idx, 'arousal', cache_s1)
    
    print("   🏋️ Training Stage 2A (High Valence)...")
    m2a, le2a = train_stage(train_idx, 'high', cache_s2a)
    
    print("   🏋️ Training Stage 2B (Low Valence)...")
    m2b, le2b = train_stage(train_idx, 'low', cache_s2b)
    
    # B. TESTING
    print("   🧪 Testing Pipeline...")
    y_pred_fold = []
    y_true_fold = []
    
    for i in tqdm(test_idx, leave=False):
        path = X_paths[i]
        true_label = y_labels[i]
        
        # 1. Predict Stage 1
        vec1 = cache_s1[path].reshape(1, -1)
        p1 = m1.predict(vec1, verbose=0)[0]
        pred_s1 = le1.inverse_transform([np.argmax(p1)])[0]
        
        final_pred = ""
        
        # 2. Routing
        if pred_s1 == 'high':
            vec2a = cache_s2a[path].reshape(1, -1)
            p2a = m2a.predict(vec2a, verbose=0)[0]
            final_pred = le2a.inverse_transform([np.argmax(p2a)])[0]
        else:
            vec2b = cache_s2b[path].reshape(1, -1)
            p2b = m2b.predict(vec2b, verbose=0)[0]
            final_pred = le2b.inverse_transform([np.argmax(p2b)])[0]
            
        y_pred_fold.append(final_pred)
        y_true_fold.append(true_label)
        
    # C. RECORD SCORE
    acc = accuracy_score(y_true_fold, y_pred_fold)
    fold_accuracies.append(acc)
    print(f"   🎯 Akurasi Fold {fold+1}: {acc*100:.2f}%")
    
    labels = ['angry', 'happy', 'sad', 'relaxed']
    cm = confusion_matrix(y_true_fold, y_pred_fold, labels=labels)
    confusion_matrices.append(cm)

# --- 5. FINAL REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR CROSS VALIDATION (RAW DATA)")
print("="*50)

avg_acc = np.mean(fold_accuracies) * 100
std_acc = np.std(fold_accuracies) * 100

print(f"📈 Rata-Rata Akurasi: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
print(f"📝 Per Fold: {[f'{x*100:.2f}%' for x in fold_accuracies]}")

# Plot Average CM
avg_cm = np.mean(confusion_matrices, axis=0).astype(int)
plt.figure(figsize=(8,6))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title(f'Average Confusion Matrix (5-Fold CV)\nRaw Data (Trimmed) - Avg Acc: {avg_acc:.2f}%')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig('cm_exp10_raw_cv.png')
plt.show()