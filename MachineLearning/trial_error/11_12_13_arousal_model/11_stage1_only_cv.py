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

# Import feature extractor yang sudah ada
import features_hierarchy as feat

# --- KONFIGURASI ---
RAW_DATA_DIR = 'data/raw' 
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI EKSPERIMEN 11: FOKUS STAGE 1 (BINARY HIGH vs LOW)...")

# --- 1. LOAD & MAPPING DATA ---
print("📥 Loading Raw Data & Mapping ke Binary...")

X_paths = []
y_binary = [] # Labelnya cuma 'high' atau 'low'

moods = ['angry', 'happy', 'sad', 'relaxed']

# Cache untuk menyimpan fitur biar training cepat
feature_cache = {}

for mood in moods:
    folder = os.path.join(RAW_DATA_DIR, mood)
    if not os.path.exists(folder): continue
        
    files = os.listdir(folder)
    for f in tqdm(files, desc=f"Load {mood}"):
        if not f.endswith(('.wav', '.mp3')): continue
        
        path = os.path.join(folder, f)
        
        # Mapping Logic
        if mood in ['angry', 'happy']:
            label = 'high'
        else:
            label = 'low'
            
        # Ekstrak Fitur Stage 1 (YAMNet + RMS + ZCR)
        # Kita cek cache dulu
        if path not in feature_cache:
            vec = feat.extract_stage_1(path)
            if vec is None: continue
            feature_cache[path] = vec
        
        X_paths.append(path)
        y_binary.append(label)

X_paths = np.array(X_paths)
y_binary = np.array(y_binary)

print(f"✅ Total Data: {len(X_paths)}")
print(f"   High Arousal: {np.sum(y_binary == 'high')}")
print(f"   Low Arousal : {np.sum(y_binary == 'low')}")

# --- 2. MODEL DEFINITION ---
def create_binary_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        # Output cuma 2 neuron (High/Low)
        Dense(2, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 3. 5-FOLD LOOP ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

fold_scores = []
confusion_matrices = []

# Encode Label (High/Low -> 0/1)
le = LabelEncoder()
y_encoded_all = le.fit_transform(y_binary)
y_categorical_all = to_categorical(y_encoded_all)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_paths, y_encoded_all)):
    print(f"\n🔄 --- FOLD {fold+1} / {FOLDS} ---")
    
    # Split Data
    X_train = np.array([feature_cache[p] for p in X_paths[train_idx]])
    y_train = y_categorical_all[train_idx]
    
    X_test = np.array([feature_cache[p] for p in X_paths[test_idx]])
    y_test = y_categorical_all[test_idx]
    
    # Class Weights (Jaga-jaga kalau imbalance)
    y_train_int = np.argmax(y_train, axis=1)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
    weights_dict = dict(enumerate(weights))
    
    # Train
    model = create_binary_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Kita pakai 20% data training sebagai validasi internal untuk early stopping
    model.fit(X_train, y_train, 
              validation_split=0.2,
              epochs=80, 
              batch_size=16, 
              callbacks=[early_stop], 
              class_weight=weights_dict, 
              verbose=0) # Silent mode
    
    # Evaluasi
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_idx = np.argmax(y_pred_prob, axis=1)
    y_true_idx = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true_idx, y_pred_idx)
    fold_scores.append(acc)
    
    print(f"   🎯 Akurasi: {acc*100:.2f}%")
    
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    confusion_matrices.append(cm)

# --- 4. FINAL REPORT ---
print("\n" + "="*50)
print("📊 HASIL STAGE 1 (HIGH vs LOW) - RAW DATA")
print("="*50)

avg_acc = np.mean(fold_scores) * 100
std_acc = np.std(fold_scores) * 100

print(f"📈 Rata-Rata Akurasi: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
print(f"📝 Per Fold: {[f'{x*100:.2f}%' for x in fold_scores]}")

# Plot Average CM
avg_cm = np.mean(confusion_matrices, axis=0).astype(int)
labels = le.classes_ # ['high', 'low'] biasanya

plt.figure(figsize=(6,5))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
plt.title(f'Binary Arousal Classification (5-Fold)\nAvg Acc: {avg_acc:.2f}%')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig('cm_exp11_stage1_binary.png')
plt.show()

# Print detail classification report dari fold terakhir (sebagai sampel)
print("\nSampel Classification Report (Fold Terakhir):")
print(classification_report(y_true_idx, y_pred_idx, target_names=labels))