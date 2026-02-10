import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Model Zoo
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Eval Tools
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
INPUT_FILE = 'gold_standard_trifecta.csv'
MODEL_SAVE_PATH = 'models/stage2b_trifecta_best_model.pkl'
SCALER_SAVE_PATH = 'models/stage2b_trifecta_scaler.pkl'
SEED = 42

# KUNCI 1: Lock Numpy Global Randomness
np.random.seed(SEED)

print("🚀 FULL BENCHMARKING: TRIFECTA EDITION")
print("   Fitur: Contrast + Sadness + Joy")
print("   Dataset: Gold Standard (40 Samples)")

# ================= 1. LOAD & LOCK DATA =================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File '{INPUT_FILE}' tidak ditemukan.")
    exit()

df = pd.read_csv(INPUT_FILE)

# KUNCI 2: Lock Data Shuffle
# Kita acak sekali lalu reset index, menjamin urutan baris selalu sama.
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Encoding Label
if 'original_label' in df.columns:
    df['label_encoded'] = df['original_label'].map({'sad': 0, 'relaxed': 1})
else:
    print("❌ Error: Kolom label tidak ditemukan.")
    exit()

# FITUR TRIFECTA
feature_cols = ['contrast', 'lyric_sad', 'lyric_joy']
print(f"\n📊 Data Loaded: {len(df)} samples")
print(f"   Features: {feature_cols}")

X = df[feature_cols].values
y = df['label_encoded'].values

# Scaling (Wajib untuk SVM/KNN/Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= 2. DEFINISI 5 MODEL =================
models = [
    # 1. Random Forest (Robust)
    ('Random Forest', RandomForestClassifier(n_estimators=50, max_depth=3, random_state=SEED)),
    
    # 2. XGBoost (Gradient Boosting)
    ('XGBoost', xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, 
                                  eval_metric='logloss', use_label_encoder=False, random_state=SEED)),
    
    # 3. SVM (Support Vector Machine) - Jago di data kecil & dimensi rendah
    ('SVM (RBF)', SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED)), 
    
    # 4. Logistic Regression (Linear) - Baseline terbaik untuk fitur yang terpisah jelas
    ('Logistic Regression', LogisticRegression(random_state=SEED)), 
    
    # 5. KNN (Distance Based)
    ('KNN (5 Neighbors)', KNeighborsClassifier(n_neighbors=5)) 
]

# ================= 3. BATTLE ARENA (CV LOOP) =================
results = []
names = []
best_score = 0
best_model = None
best_name = ""

print("\n" + "="*100)
print(f"{'MODEL':<20} | {'FOLD 1':<8} | {'FOLD 2':<8} | {'FOLD 3':<8} | {'FOLD 4':<8} | {'FOLD 5':<8} | {'MEAN':<8} | {'STD':<8}")
print("="*100)

for name, model in models:
    # StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Cross Validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    
    results.append(cv_scores)
    names.append(name)
    
    # Formatting Output
    folds_str = " | ".join([f"{s*100:.0f}%" for s in cv_scores]) # Dibulatkan biar rapi
    mean_score = np.mean(cv_scores) * 100
    std_dev = np.std(cv_scores) * 100
    
    print(f"{name:<20} | {folds_str} | {mean_score:.2f}%   | ±{std_dev:.2f}%")
    
    # Cari Juara
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_name = name
    elif mean_score == best_score:
        # Jika seri, pilih yang Std Dev-nya lebih kecil (lebih stabil)
        current_best_std = np.std(cross_val_score(best_model, X_scaled, y, cv=skf)) * 100
        if std_dev < current_best_std:
            best_model = model
            best_name = name

print("="*100)
print(f"\n🏆 JUARA UMUM: {best_name.upper()} (Akurasi: {best_score:.2f}%)")

# ================= 4. SAVE THE CHAMPION =================
if not os.path.exists('models'): os.makedirs('models')

print(f"\n💾 Menyimpan model terbaik ({best_name})...")
best_model.fit(X_scaled, y) # Train ulang pada seluruh data

joblib.dump(best_model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

# Info File
with open('models/stage2b_info.txt', 'w') as f:
    f.write(f"Model: {best_name} (Best of 5 Benchmark)\n")
    f.write(f"Dataset: Gold Standard Trifecta\n")
    f.write(f"Features: {feature_cols}\n")
    f.write(f"CV Accuracy: {best_score:.2f}%\n")

print(f"✅ Model Tersimpan: {MODEL_SAVE_PATH}")
print(f"✅ Scaler Tersimpan: {SCALER_SAVE_PATH}")

# ================= 5. VISUALISASI PERBANDINGAN =================
plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=names, patch_artist=True, showmeans=True)
plt.title(f'Benchmarking 5 Model (Trifecta Features)\nWinner: {best_name}')
plt.ylabel('Akurasi Cross-Validation')
plt.grid(True, axis='y', alpha=0.3)

# Warna Pastel
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#E0E0E0']
for patch, color in zip(plt.gca().artists, colors):
    patch.set_facecolor(color)

plt.tight_layout()
plt.show()