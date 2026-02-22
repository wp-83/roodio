import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix

# ==============================================================================
# WAJIB DITAMBAHKAN SESUAI PROMPT
# ==============================================================================
mlflow.set_experiment("Baseline_Text_Only")

# ==============================================================================
# 1. PERSIAPAN DATA (ROBERTA LYRICS PROBABILITIES)
# ==============================================================================
print("[1/3] Memuat Dataset Lirik (RoBERTa)...")

# --- SIMULASI DATA (Hapus bagian ini jika sudah punya file CSV asli) ---
np.random.seed(42)
n_samples = 400
moods = ['Angry', 'Happy', 'Sad', 'Relaxed']

data = {
    'prob_angry': np.random.rand(n_samples),
    'prob_happy': np.random.rand(n_samples),
    'prob_sad': np.random.rand(n_samples),
    'prob_relaxed': np.random.rand(n_samples),
    'Label': np.random.choice(moods, n_samples)
}
df = pd.DataFrame(data)
# Normalisasi probabilitas agar totalnya 1 per baris
probs = df[['prob_angry', 'prob_happy', 'prob_sad', 'prob_relaxed']].values
df[['prob_angry', 'prob_happy', 'prob_sad', 'prob_relaxed']] = probs / probs.sum(axis=1, keepdims=True)
# -----------------------------------------------------------------------

X = df[['prob_angry', 'prob_happy', 'prob_sad', 'prob_relaxed']]
y = df['Label']

# ==============================================================================
# 2. DEFINISI MODEL & CROSS-VALIDATION
# ==============================================================================
print("[2/3] Memulai Eksperimen MLflow & 5-Fold CV...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ==============================================================================
# 3. TRAINING & LOGGING KE MLFLOW
# ==============================================================================
if not os.path.exists("plots"):
    os.makedirs("plots")

for model_name, model in models.items():
    print(f"\nTraining model: {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        # 1. Log Parameter Model
        mlflow.log_param("model_type", model_name)
        if model_name == "Random Forest":
            mlflow.log_param("n_estimators", 100)
        
        # 2. Lakukan 5-Fold CV (Untuk Metrik)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mean_acc = cv_scores.mean()
        std_acc = cv_scores.std()
        
        # Log Metrik ke MLflow
        mlflow.log_metric("cv_accuracy_mean", mean_acc)
        mlflow.log_metric("cv_accuracy_std", std_acc)
        print(f"   --> Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
        
        # 3. Prediksi & Confusion Matrix
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, labels=moods)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=moods, yticklabels=moods)
        plt.title(f'Confusion Matrix: {model_name}\n(5-Fold CV)')
        plt.ylabel('Actual Mood')
        plt.xlabel('Predicted Mood')
        plt.tight_layout()
        
        plot_filename = f"plots/CM_{model_name.replace(' ', '_')}.png"
        plt.savefig(plot_filename)
        plt.close()
        
        # 4. Log Gambar ke MLflow
        mlflow.log_artifact(plot_filename)
print("\n[3/3] Selesai! Semua metrik dan gambar telah dilog ke MLflow.")
