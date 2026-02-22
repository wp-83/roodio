import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
INPUT_FILE = 'gold_standard_data.csv'
MODEL_SAVE_PATH = 'models/stage2b_gold_xgb.pkl'
SCALER_SAVE_PATH = 'models/stage2b_gold_scaler.pkl'
SEED = 42

print("🚀 TRAINING STAGE 2B: GOLD STANDARD EDITION")
print("   Menggunakan 40 Data Terbaik (Archetypes) untuk melatih 'Guru Besar' AI.")

# ================= 1. LOAD DATA =================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File '{INPUT_FILE}' tidak ditemukan. Jalankan script mining sebelumnya dulu!")
    exit()

df = pd.read_csv(INPUT_FILE)

# Pastikan labelnya benar
# Kita asumsikan di CSV kolom label bernama 'original_label'
# 0 = Sad, 1 = Relaxed
df['label_encoded'] = df['original_label'].map({'sad': 0, 'relaxed': 1})

# FITUR PILIHAN (Berdasarkan Analisis Visual Terakhir)
# Kita pakai 4 Fitur Kunci:
# 1. rms_std (Dinamika Volume)
# 2. contrast (Tekstur Suara)
# 3. lyric_sad (Skor Sedih)
# 4. lyric_joy (Skor Happy/Relaxed)
feature_cols = ['rms_std', 'contrast', 'lyric_sad', 'lyric_joy']

X = df[feature_cols].values
y = df['label_encoded'].values

print(f"\n📊 Data Loaded: {len(df)} samples")
print(f"   - Sad    : {len(df[df['label_encoded']==0])}")
print(f"   - Relaxed: {len(df[df['label_encoded']==1])}")

# ================= 2. PREPROCESSING =================
print("\n⚖️ Scaling Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= 3. TRAINING XGBOOST =================
print("\n🔥 Training XGBoost on Gold Data...")

# Konfigurasi Model untuk "Small Data"
# Kita batasi kedalaman (depth) agar tidak menghafal mati, tapi belajar pola
clf = xgb.XGBClassifier(
    n_estimators=50,    # Cukup 50 pohon karena datanya sedikit
    learning_rate=0.1,  # Belajar standar
    max_depth=2,        # Pohon pendek (biar generalisasinya bagus)
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=SEED
)

# Cross Validation (Sangat Penting untuk Data Kecil)
# Kita pakai 5-Fold. Karena total data 40, berarti 8 data per fold.
cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

print("\n" + "="*50)
print(f"🏆 GOLD MODEL PERFORMANCE (CV Score)")
print("="*50)
print(f"✅ Mean Accuracy : {np.mean(cv_scores)*100:.2f}%")
print(f"📉 Std Dev       : ±{np.std(cv_scores)*100:.2f}%")
print("-" * 50)

# Train Final pada Seluruh 40 Data
clf.fit(X_scaled, y)
y_pred = clf.predict(X_scaled)

# Cek apakah dia berhasil "lulus ujian" dari buku teksnya sendiri
train_acc = accuracy_score(y, y_pred)
print(f"Training Accuracy (Memorization): {train_acc*100:.2f}%")

if train_acc < 0.9:
    print("⚠️ Warning: Model bahkan kesulitan mempelajari data terbaik. Cek fitur lagi.")
else:
    print("✅ Model berhasil menguasai materi 'Gold Standard' dengan baik.")

# ================= 4. SAVE MODEL =================
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(clf, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

# Simpan Info
with open('models/stage2b_info.txt', 'w') as f:
    f.write(f"Model: XGBoost Gold Standard (Trained on 40 Best Songs)\n")
    f.write(f"Features: {feature_cols}\n")
    f.write(f"CV Accuracy: {np.mean(cv_scores)*100:.2f}%\n")

print(f"\n💾 Model Tersimpan: {MODEL_SAVE_PATH}")
print(f"💾 Scaler Tersimpan: {SCALER_SAVE_PATH}")

# ================= 5. CONFUSION MATRIX VISUALIZATION =================
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sad', 'Relaxed'], yticklabels=['Sad', 'Relaxed'])
plt.title(f'Gold Model CM (Training Data)\nAcc: {train_acc*100:.1f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()