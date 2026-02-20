import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# WAJIB DITAMBAHKAN SESUAI PROMPT
# ==============================================================================
mlflow.set_experiment("Baseline_Text_Only")

print("🚀 Memulai eksperimen Text-Based Baseline...")

# ==============================================================================
# 1. LOAD DATASET YANG SUDAH BERSIH & PUNYA PROBABILITAS ROBERTA
# ==============================================================================
# Kita pakai file hasil cucian yang sudah ada angka probabilitasnya
file_path = r"C:\Users\yoyad\Documents\BINUS\CAWU FOUR\project\roodio\MachineLearning\comparisson_model\modeling\CLEAN_ROBERTA_DATA.csv"
df = pd.read_csv(file_path)

# Input HANYA probabilitas emosi dari lirik
X = df[['prob_angry', 'prob_happy', 'prob_relaxed', 'prob_sad']]
# Target klasifikasi musik (4 mood)
y = df['Label'].astype(str).str.capitalize()

# ==============================================================================
# 2. DEFINISI 3 MODEL & 5-FOLD CV
# ==============================================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
moods = sorted(y.unique().tolist())

# ==============================================================================
# 3. TRAINING, EVALUASI, DAN LOGGING KE MLFLOW
# ==============================================================================
for model_name, model in models.items():
    print(f"-> Proses Training: {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        # Catat Parameter
        mlflow.log_param("model_type", model_name)
        
        # 5-Fold CV & Catat Metrik Akurasi
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mlflow.log_metric("cv_accuracy_mean", acc_scores.mean())
        
        # Prediksi untuk Confusion Matrix
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, labels=moods)
        
        # Buat Gambar Confusion Matrix (.png)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=moods, yticklabels=moods)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual Mood')
        plt.xlabel('Predicted Mood')
        plt.tight_layout()
        
        # Simpan sementara di lokal
        img_filename = f"CM_{model_name.replace(' ', '_')}.png"
        plt.savefig(img_filename)
        plt.close()
        
        # Log Gambar ke MLflow
        mlflow.log_artifact(img_filename)

print("\n✅ EKSPERIMEN SELESAI! Semua model, metrik, dan CM berhasil dilog.")