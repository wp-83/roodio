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
from sklearn.metrics import accuracy_score

# ================= CONFIG =================
INPUT_FILE = 'gold_standard_data.csv'
SEED = 42

print("🚀 BENCHMARKING 5 MODEL (STAGE 2B GOLD STANDARD)")
print("   Mencari algoritma terbaik untuk data kecil berkualitas tinggi.")

# ================= 1. LOAD & PREPARE DATA =================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File '{INPUT_FILE}' tidak ditemukan.")
    exit()

df = pd.read_csv(INPUT_FILE)

# Encoding Label (0=Sad, 1=Relaxed)
# Sesuaikan string label dengan data CSV kamu
if 'original_label' in df.columns:
    df['label_encoded'] = df['original_label'].map({'sad': 0, 'relaxed': 1})
else:
    # Fallback jika nama kolom beda
    print("⚠️ Warning: Kolom 'original_label' tidak ditemukan, mencoba menebak...")
    # Sesuaikan logika ini jika CSV kamu beda format
    pass

# Fitur Gold Standard
feature_cols = ['rms_std', 'contrast', 'lyric_sad', 'lyric_joy']
X = df[feature_cols].values
y = df['label_encoded'].values

print(f"\n📊 Data Loaded: {len(X)} samples")

# Scaling (PENTING untuk SVM, KNN, dan Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= 2. DEFINISI 5 MODEL =================
# Kita set parameter konservatif agar tidak overfit pada data kecil
models = [
    ('Random Forest', RandomForestClassifier(n_estimators=50, max_depth=3, random_state=SEED)),
    ('XGBoost', xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, eval_metric='logloss', use_label_encoder=False, random_state=SEED)),
    ('SVM (RBF)', SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED)), # SVM jago di data kecil
    ('Logistic Regression', LogisticRegression(random_state=SEED)), # Baseline linear
    ('KNN (5 Neighbors)', KNeighborsClassifier(n_neighbors=5)) # Distance based
]

# ================= 3. TRAINING & EVALUATION LOOP =================
results = []
names = []
final_scores = []

print("\n⚔️  BATTLE OF ALGORITHMS (5-Fold CV)  ⚔️")
print("-" * 65)
print(f"{'MODEL':<20} | {'MEAN ACCURACY':<15} | {'STD DEV (Stabil)':<15}")
print("-" * 65)

best_score = 0
best_model = None
best_name = ""

for name, model in models:
    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    
    results.append(cv_scores)
    names.append(name)
    final_scores.append(mean_acc)
    
    print(f"{name:<20} | {mean_acc*100:.2f}%          | ±{std_acc*100:.2f}%")
    
    # Cek Juara
    if mean_acc > best_score:
        best_score = mean_acc
        best_model = model
        best_name = name

print("-" * 65)
print(f"\n🏆 JUARA SAAT INI: {best_name.upper()} ({best_score*100:.2f}%)")

# ================= 4. VISUALISASI =================
plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=names, patch_artist=True)
plt.title(f'Perbandingan Akurasi 5 Model (Gold Data)\nJuara: {best_name}')
plt.ylabel('Akurasi CV')
plt.grid(True, axis='y', alpha=0.3)
plt.savefig('benchmark_results.png')
print("📊 Grafik perbandingan disimpan ke 'benchmark_results.png'")
plt.show()

# ================= 5. SAVE THE WINNER =================
print(f"\n💾 Menyimpan Model Terbaik ({best_name})...")

# Train ulang model terbaik pada SELURUH data (40 sampel)
best_model.fit(X_scaled, y)

# Simpan
if not os.path.exists('models'): os.makedirs('models')
save_path_model = 'models/stage2b_best_gold_model.pkl'
save_path_scaler = 'models/stage2b_best_gold_scaler.pkl'

joblib.dump(best_model, save_path_model)
joblib.dump(scaler, save_path_scaler)

# Simpan Info Text
with open('models/stage2b_info.txt', 'w') as f:
    f.write(f"Model Terbaik: {best_name}\n")
    f.write(f"CV Accuracy: {best_score*100:.2f}%\n")
    f.write(f"Features: {feature_cols}\n")
    f.write(f"Trained on: 40 Gold Standard Samples\n")

print(f"✅ Model Tersimpan: {save_path_model}")
print(f"✅ Scaler Tersimpan: {save_path_scaler}")
print("\n👉 Gunakan model ini di aplikasi final kamu!")