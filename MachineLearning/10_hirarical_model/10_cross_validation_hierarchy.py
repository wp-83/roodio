import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import feature extractor kita
import features_hierarchy as feat

# --- KONFIGURASI ---
# Gunakan folder data processed yang paling lengkap (gabungan train+test kalau bisa)
# Atau kita load dari split_exp4 dan gabung manual di memori
DATA_ROOT = 'data/processed_exp4' 
FOLDS = 5
SEED = 42

print(f"🚀 MEMULAI {FOLDS}-FOLD CROSS VALIDATION HIERARCHICAL...")

# --- 1. LOAD ALL DATA INTO MEMORY ---
# Kita butuh X (fitur) dan y (label) untuk seluruh data
X_paths = []
y_labels = []

print("📂 Loading seluruh data ke memori...")
for subset in ['train', 'test']:
    subset_dir = os.path.join(DATA_ROOT, subset)
    if not os.path.exists(subset_dir): continue
    
    for mood in ['angry', 'happy', 'sad', 'relaxed']:
        mood_dir = os.path.join(subset_dir, mood)
        if not os.path.exists(mood_dir): continue
        
        files = os.listdir(mood_dir)
        for f in tqdm(files, desc=f"Load {subset}/{mood}", leave=False):
            if f.endswith('.wav') or f.endswith('.mp3'):
                X_paths.append(os.path.join(mood_dir, f))
                y_labels.append(mood)

X_paths = np.array(X_paths)
y_labels = np.array(y_labels)

print(f"✅ Total Data: {len(X_paths)} file.")

# --- 2. HELPER FUNCTION: CREATE MODEL ---
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

# --- 3. HELPER: TRAINING SATU STAGE ---
def train_stage_fold(X_train_paths, y_train_labels, stage_type, extract_func):
    """
    Melatih model untuk stage tertentu pada fold tertentu.
    stage_type: 'arousal', 'high_val', 'low_val'
    """
    # Filter data sesuai stage
    X_vec, y_target = [], []
    
    # Logic filtering data training
    for path, label in zip(X_train_paths, y_train_labels):
        # STAGE 1: Pakai SEMUA data, labelnya diubah jadi high/low
        if stage_type == 'arousal':
            target = 'high' if label in ['angry', 'happy'] else 'low'
            vec = extract_func(path)
            X_vec.append(vec)
            y_target.append(target)
            
        # STAGE 2A: Cuma pakai data Angry/Happy
        elif stage_type == 'high_val':
            if label in ['angry', 'happy']:
                vec = extract_func(path)
                X_vec.append(vec)
                y_target.append(label)
                
        # STAGE 2B: Cuma pakai data Sad/Relaxed
        elif stage_type == 'low_val':
            if label in ['sad', 'relaxed']:
                vec = extract_func(path)
                X_vec.append(vec)
                y_target.append(label)
    
    X_vec = np.array(X_vec)
    
    # Encode Labels
    le = LabelEncoder()
    y_enc = to_categorical(le.fit_transform(y_target))
    
    # Class Weights
    y_int = np.argmax(y_enc, axis=1)
    if len(np.unique(y_int)) > 1: # Cek biar gak error kalau cuma 1 kelas
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
        weights_dict = dict(enumerate(weights))
    else:
        weights_dict = None

    # Train Model
    model = create_model(X_vec.shape[1], 2)
    early_stop = EarlyStopping(patience=10, restore_best_weights=True, monitor='loss')
    
    # Silent training (verbose=0) biar gak menuhin layar
    model.fit(X_vec, y_enc, epochs=60, batch_size=16, 
              callbacks=[early_stop], class_weight=weights_dict, verbose=0)
    
    return model, le

# --- 4. START K-FOLD LOOP ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

fold_accuracies = []
confusion_matrices = []

# Kita pakai StratifiedKFold biar sebaran mood tiap fold rata
for fold, (train_idx, test_idx) in enumerate(skf.split(X_paths, y_labels)):
    print(f"\n🔄 --- FOLD {fold+1} / {FOLDS} ---")
    
    # Split Data
    X_train_p, X_test_p = X_paths[train_idx], X_paths[test_idx]
    y_train_l, y_test_l = y_labels[train_idx], y_labels[test_idx]
    
    # A. TRAIN 3 MODEL TERPISAH
    print("   🏋️ Training Stage 1 (Arousal)...")
    m1, le1 = train_stage_fold(X_train_p, y_train_l, 'arousal', feat.extract_stage_1)
    
    print("   🏋️ Training Stage 2A (High Valence)...")
    m2a, le2a = train_stage_fold(X_train_p, y_train_l, 'high_val', feat.extract_stage_2a)
    
    print("   🏋️ Training Stage 2B (Low Valence)...")
    m2b, le2b = train_stage_fold(X_train_p, y_train_l, 'low_val', feat.extract_stage_2b)
    
    # B. TESTING HIERARCHY PIPELINE
    print("   🧪 Testing Pipeline...")
    y_pred_fold = []
    y_true_fold = []
    
    for path, true_label in tqdm(zip(X_test_p, y_test_l), total=len(X_test_p), leave=False):
        try:
            # 1. Predict Stage 1
            vec1 = feat.extract_stage_1(path).reshape(1, -1)
            p1 = m1.predict(vec1, verbose=0)[0]
            pred_s1 = le1.inverse_transform([np.argmax(p1)])[0]
            
            final_pred = ""
            
            # 2. Routing
            if pred_s1 == 'high':
                vec2a = feat.extract_stage_2a(path).reshape(1, -1)
                p2a = m2a.predict(vec2a, verbose=0)[0]
                final_pred = le2a.inverse_transform([np.argmax(p2a)])[0]
            else:
                # Perhatikan: di fold tertentu, 'low' itu dipetakan ke le1 kelas index berapa
                # Kita asumsikan mappingnya konsisten
                vec2b = feat.extract_stage_2b(path).reshape(1, -1)
                p2b = m2b.predict(vec2b, verbose=0)[0]
                final_pred = le2b.inverse_transform([np.argmax(p2b)])[0]
            
            y_pred_fold.append(final_pred)
            y_true_fold.append(true_label)
            
        except Exception as e:
            print(f"Error {path}: {e}")
    
    # C. RECORD SCORE FOLD INI
    acc = accuracy_score(y_true_fold, y_pred_fold)
    fold_accuracies.append(acc)
    print(f"   🎯 Akurasi Fold {fold+1}: {acc*100:.2f}%")
    
    # Simpan CM untuk dirata-rata nanti
    labels = ['angry', 'happy', 'sad', 'relaxed']
    cm = confusion_matrix(y_true_fold, y_pred_fold, labels=labels)
    confusion_matrices.append(cm)

# --- 5. FINAL REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR CROSS VALIDATION (5-FOLD)")
print("="*50)

avg_acc = np.mean(fold_accuracies) * 100
std_acc = np.std(fold_accuracies) * 100

print(f"📈 Rata-Rata Akurasi: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
print(f"📝 Detail per Fold: {[f'{x*100:.2f}%' for x in fold_accuracies]}")

# Plot Average Confusion Matrix
avg_cm = np.mean(confusion_matrices, axis=0).astype(int)
plt.figure(figsize=(8,6))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title(f'Average Confusion Matrix ({FOLDS}-Fold CV)\nAvg Acc: {avg_acc:.2f}%')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig('cm_exp10_kfold_average.png')
plt.show()

print("\n✅ SELESAI! Gambar matrix rata-rata disimpan.")