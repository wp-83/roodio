import os
import numpy as np
import pandas as pd
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
INPUT_FILE = 'gold_standard_data.csv'
SEED = 43  # Angka keberuntungan

# KUNCI 1: Mengunci Randomness Global Numpy
np.random.seed(SEED)

print("🚀 FULL BENCHMARKING REPORT (LOCKED VERSION)")
print("   Hasil dipastikan stabil dan tidak berubah-ubah.")

# ================= 1. LOAD DATA =================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File '{INPUT_FILE}' tidak ditemukan.")
    exit()

df = pd.read_csv(INPUT_FILE)

# KUNCI 2: Mengunci Urutan Data Sebelum Masuk Model
# Kita acak data sekali dengan seed, lalu reset indexnya.
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Encoding Label
if 'original_label' in df.columns:
    df['label_encoded'] = df['original_label'].map({'sad': 0, 'relaxed': 1})
    print(f"📊 Data Loaded: {len(df)} samples")
else:
    print("❌ Error: Kolom label tidak ada.")
    exit()

# Fitur Gold Standard
feature_cols = ['rms_std', 'contrast', 'lyric_sad', 'lyric_joy']
X = df[feature_cols].values
y = df['label_encoded'].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= 2. DEFINISI MODEL =================
models = [
    ('Random Forest', RandomForestClassifier(n_estimators=50, max_depth=3, random_state=SEED)),
    ('XGBoost', xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, eval_metric='logloss', use_label_encoder=False, random_state=SEED)),
    ('SVM (RBF)', SVC(kernel='rbf', C=1.0, random_state=SEED)), 
    ('Logistic Regression', LogisticRegression(random_state=SEED)), 
    ('KNN (5 Neighbors)', KNeighborsClassifier(n_neighbors=5)) 
]

# ================= 3. EVALUATION LOOP =================
results = []
names = []

print("\n" + "="*100)
print(f"{'MODEL':<20} | {'FOLD 1':<8} | {'FOLD 2':<8} | {'FOLD 3':<8} | {'FOLD 4':<8} | {'FOLD 5':<8} | {'MEAN':<8} | {'STD DEV':<8}")
print("="*100)

for name, model in models:
    # StratifiedKFold dengan Seed yang sama
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    
    results.append(cv_scores)
    names.append(name)
    
    folds_str = " | ".join([f"{s*100:.1f}%" for s in cv_scores])
    mean_score = np.mean(cv_scores) * 100
    std_dev = np.std(cv_scores) * 100
    
    print(f"{name:<20} | {folds_str} | {mean_score:.2f}%   | ±{std_dev:.2f}%")

print("="*100)

# ================= 4. VISUALISASI =================
plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=names, patch_artist=True, showmeans=True)
plt.title(f'Benchmark Result (SEED={SEED})')
plt.ylabel('Accuracy')
plt.grid(True, axis='y', alpha=0.3)
colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'lightgrey']
for patch, color in zip(plt.gca().artists, colors):
    patch.set_facecolor(color)
plt.show()