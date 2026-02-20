import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. WAJIB DITAMBAHKAN SESUAI PROMPT
mlflow.set_experiment("Baseline_Text_Only")

print("Memulai eksperimen MLflow untuk lirik dengan dataset asli...")

# ==============================================================================
# 2. LOAD DATASET ASLI (MENGGUNAKAN ABSOLUTE PATH)
# ==============================================================================
# Gunakan huruf 'r' di depan tanda kutip agar Windows membaca tanda '\' dengan benar
file_path = r"C:\Users\yoyad\Documents\BINUS\CAWU FOUR\project\roodio\MachineLearning\comparisson_model\modeling\lyrics.xlsx"

try:
    # Coba baca sebagai file Excel
    df = pd.read_excel(file_path)
except Exception:
    # Fallback: Jika gagal, paksa baca sebagai CSV
    df = pd.read_csv(file_path)

# --- SISTEM DETEKSI ERROR OTOMATIS ---
print("Kolom yang terdeteksi di datamu:", df.columns.tolist())

# Mengecek apakah kolom probabilitas RoBERTa sudah ada di file
try:
    X = df[['prob_angry', 'prob_happy', 'prob_relaxed', 'prob_sad']]
except KeyError:
    print("\n❌ STOP! ERROR FATAL: Kolom angka probabilitas tidak ditemukan di Excel-mu.")
    print("Sepertinya ini masih file lirik mentah. Sesuai prompt tugasmu, inputnya HARUS probabilitas emosi dari RoBERTa.")
    print("Silakan jalankan liriknya ke model RoBERTa dulu, atau masukkan file CSV/Excel yang sudah berisi angka probabilitas!")
    sys.exit()

# Sesuai cuplikan datamu, nama kolom targetnya adalah huruf kecil 'mood'
try:
    y = df['mood'] 
except KeyError:
    y = df['Label'] # Fallback kalau ternyata namanya Label
# ==============================================================================

# 3. SIAPKAN 3 MODEL ML
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Siapkan 5-Fold Cross Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
moods = ['angry', 'happy', 'relaxed', 'sad'] # Huruf kecil menyesuaikan dataset asli

# 4. MULAI TRAINING DAN LOGGING KE MLFLOW
for name, model in models.items():
    print(f"-> Training {name}...")
    
    with mlflow.start_run(run_name=name):
        # Catat nama model
        mlflow.log_param("model_name", name)
        
        # Hitung Akurasi dengan 5-Fold CV
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mlflow.log_metric("accuracy_mean", acc_scores.mean())
        
        # Prediksi untuk membuat Confusion Matrix
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, labels=moods)
        
        # Buat Gambar Confusion Matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=moods, yticklabels=moods)
        plt.title(f"Confusion Matrix: {name}")
        plt.ylabel('Label Asli')
        plt.xlabel('Tebakan Model')
        plt.tight_layout()
        
        # Simpan gambar .png
        img_filename = f"cm_{name.replace(' ', '_')}.png"
        plt.savefig(img_filename)
        plt.close()
        
        # Log gambar .png tersebut ke dalam MLflow
        mlflow.log_artifact(img_filename)

print("✅ Selesai! Ketik 'mlflow ui' di terminal untuk melihat hasil.")