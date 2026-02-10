import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
INPUT_FILE = 'gold_standard_trifecta.csv'
MODEL_SAVE_PATH = 'models/stage2b_trifecta_xgb.pkl'
SCALER_SAVE_PATH = 'models/stage2b_trifecta_scaler.pkl'
SEED = 42

print("🚀 TRAINING STAGE 2B: TRIFECTA FINAL")
print("   Fitur: Contrast (Audio) + Sadness (Text) + Joy (Text)")
print("   Dataset: Gold Standard (40 Best Samples)")

# ================= 1. LOAD DATA =================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File '{INPUT_FILE}' tidak ditemukan. Jalankan script mining trifecta dulu!")
    exit()

df = pd.read_csv(INPUT_FILE)

# Encoding Label (0=Sad, 1=Relaxed)
if 'original_label' in df.columns:
    df['label_encoded'] = df['original_label'].map({'sad': 0, 'relaxed': 1})
else:
    print("❌ Error: Kolom label tidak ditemukan.")
    exit()

# --- PILIHAN FITUR FINAL (HANYA 3) ---
feature_cols = ['contrast', 'lyric_sad', 'lyric_joy']

print(f"\n📊 Data Loaded: {len(df)} samples")
print(f"   Fitur yang dipakai: {feature_cols}")

X = df[feature_cols].values
y = df['label_encoded'].values

# ================= 2. PREPROCESSING =================
print("\n⚖️  Scaling Features...")
# Scaling SANGAT PENTING karena 'contrast' (satuan 20-an) beda jauh dengan 'lyric' (satuan 0-1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= 3. TRAINING XGBOOST =================
print("\n🔥 Training XGBoost Model...")

# Konfigurasi untuk Data Kecil & Berkualitas
clf = xgb.XGBClassifier(
    n_estimators=50,    # Cukup 50 pohon
    max_depth=2,        # Pohon pendek (Depth 2 cukup untuk memisah 3 fitur)
    learning_rate=0.1,  # Belajar standar
    subsample=0.8,      # Sedikit variasi data
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=SEED
)

# Cross Validation (Cek Kestabilan)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='accuracy')

print("\n" + "="*50)
print(f"🏆 TRIFECTA MODEL PERFORMANCE")
print("="*50)
print(f"✅ CV Mean Accuracy : {np.mean(cv_scores)*100:.2f}%")
print(f"📉 Std Deviation    : ±{np.std(cv_scores)*100:.2f}%")
print("-" * 50)

# Final Fit (Latih pada seluruh 40 data)
clf.fit(X_scaled, y)
y_pred = clf.predict(X_scaled)
train_acc = accuracy_score(y, y_pred)

print(f"Training Accuracy   : {train_acc*100:.2f}% (Memorization)")

if train_acc < 0.95:
    print("⚠️ Warning: Model belum menghafal data gold dengan sempurna.")
else:
    print("✅ Model SIAP DIGUNAKAN! (Sangat Akurat pada Data Gold)")

# ================= 4. SAVE MODEL & SCALER =================
if not os.path.exists('models'): os.makedirs('models')

joblib.dump(clf, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

# Simpan Info Text
with open('models/stage2b_info.txt', 'w') as f:
    f.write("Model: Stage 2B Trifecta (XGBoost)\n")
    f.write(f"Features: {feature_cols}\n")
    f.write(f"Accuracy (CV): {np.mean(cv_scores)*100:.2f}%\n")
    f.write("Note: Trained on Gold Standard Trifecta Data\n")

print(f"\n💾 Model Tersimpan: {MODEL_SAVE_PATH}")
print(f"💾 Scaler Tersimpan: {SCALER_SAVE_PATH}")

# ================= 5. VISUALISASI =================
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sad', 'Relaxed'], yticklabels=['Sad', 'Relaxed'])
plt.title(f'Trifecta Model CM\nAcc: {train_acc*100:.1f}%')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()