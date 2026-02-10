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

# Tuning & Eval Tools
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
INPUT_FILE = 'gold_standard_trifecta.csv'
MODEL_SAVE_PATH = 'models/stage2b_tuned_best.pkl'
SCALER_SAVE_PATH = 'models/stage2b_tuned_scaler.pkl'
SEED = 42

# Kunci Randomness
np.random.seed(SEED)

print("🚀 HYPERPARAMETER TUNING (GRID SEARCH)")
print("   Mencari model yang 'Jujur' dan Tahan Banting (Anti-Overfit).")

# ================= 1. LOAD DATA =================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: {INPUT_FILE} not found.")
    exit()

df = pd.read_csv(INPUT_FILE)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True) # Shuffle lock

if 'original_label' in df.columns:
    df['label_encoded'] = df['original_label'].map({'sad': 0, 'relaxed': 1})

X = df[['contrast', 'lyric_sad', 'lyric_joy']].values
y = df['label_encoded'].values

print(f"📊 Data: {len(X)} samples. Features: 3 (Contrast, Sad, Joy)")

# Scaling (WAJIB untuk SVM/KNN/LogReg)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= 2. DEFINISI GRID PARAMETER =================
# Ini adalah "Menu" yang akan dicoba satu per satu oleh komputer
model_params = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=SEED),
        'params': {
            'n_estimators': [30, 50, 100],
            'max_depth': [2, 3, 4],       # KUNCI ANTI OVERFIT: Batasi kedalaman pohon!
            'min_samples_leaf': [2, 4],   # Minimal sampel di daun (biar gak terlalu spesifik)
            'criterion': ['gini', 'entropy']
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED),
        'params': {
            'n_estimators': [30, 50],
            'max_depth': [2, 3],          # Pohon pendek = Generalisasi lebih baik
            'learning_rate': [0.05, 0.1],
            'reg_lambda': [1, 10],        # L2 Regularization (Rem biar gak ngebut)
            'subsample': [0.8]
        }
    },
    'SVM': {
        'model': SVC(random_state=SEED, probability=True),
        'params': {
            'C': [0.1, 1, 10],            # C Kecil = Margin Lebar (Lebih aman dari overfit)
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=SEED),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],  # Coba tetangga lebih banyak biar lebih smooth
            'weights': ['uniform', 'distance']
        }
    }
}

# ================= 3. TUNING LOOP =================
best_global_score = 0
best_global_model = None
best_global_name = ""
results_summary = []

print("\n" + "="*80)
print(f"{'MODEL':<20} | {'BEST PARAMS (ANTI-OVERFIT)':<40} | {'ACCURACY':<8}")
print("="*80)

for name, mp in model_params.items():
    # Grid Search dengan 5-Fold CV
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_scaled, y)
    
    best_score = clf.best_score_
    best_params = clf.best_params_
    
    # Simpan summary
    results_summary.append({
        'Model': name,
        'Accuracy': best_score,
        'Best_Params': str(best_params)
    })
    
    # Format string parameter biar rapi
    param_str = str(best_params).replace('{', '').replace('}', '').replace("'", "")
    if len(param_str) > 40: param_str = param_str[:37] + "..."
    
    print(f"{name:<20} | {param_str:<40} | {best_score*100:.2f}%")
    
    # Cek Juara Global
    if best_score > best_global_score:
        best_global_score = best_score
        best_global_model = clf.best_estimator_
        best_global_name = name

print("="*80)

# ================= 4. VALIDASI STABILITAS JUARA =================
print(f"\n🏆 JUARA TUNING: {best_global_name.upper()} ({best_global_score*100:.2f}%)")
print("   Sekarang kita cek kestabilannya per-fold...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(best_global_model, X_scaled, y, cv=skf, scoring='accuracy')

print(f"\n   Fold Details: {cv_scores}")
print(f"   MEAN: {np.mean(cv_scores)*100:.2f}%")
print(f"   STD : ±{np.std(cv_scores)*100:.2f}%")

if np.std(cv_scores) > 0.10:
    print("\n⚠️ WARNING: Model ini masih agak labil (Std Dev > 10%).")
    print("   Saran: Jika ada model lain dengan akurasi mirip tapi Std Dev lebih kecil, pilih itu.")
else:
    print("\n✅ STATUS: Model Stabil & Terpercaya.")

# ================= 5. SAVE =================
print(f"\n💾 Menyimpan Model Terbaik ({best_global_name})...")
best_global_model.fit(X_scaled, y) # Train full
joblib.dump(best_global_model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

# Info File
with open('models/stage2b_info.txt', 'w') as f:
    f.write(f"Model: {best_global_name} (Tuned)\n")
    f.write(f"Best Params: {best_global_model.get_params()}\n")
    f.write(f"Accuracy: {best_global_score*100:.2f}%\n")

print(f"✅ Selesai! Gunakan '{MODEL_SAVE_PATH}' untuk aplikasi final.")