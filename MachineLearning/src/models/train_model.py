import pandas as pd
import numpy as np
import os
import argparse
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Algoritma & Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Tuning & Utilities
from sklearn.model_selection import train_test_split, GridSearchCV # <-- Senjata Baru
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- KONFIGURASI PATH ---
PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = os.path.join(PROJECT_DIR, 'data', 'processed', 'features.csv')
REPORT_PATH = os.path.join(PROJECT_DIR, 'reports')
os.makedirs(REPORT_PATH, exist_ok=True)

def train(model_type):
    # 1. SETUP MLFLOW
    # Kita ganti nama eksperimen untuk membedakan hasil Tuning vs Manual
    mlflow.set_experiment("Roodio_Mood_Tuning_V1")
    
    print(f"\n{'='*60}")
    print(f"TRAINING + TUNING (GridSearch): {model_type.upper()}")
    print(f"{'='*60}")

    # 2. LOAD DATA
    if not os.path.exists(INPUT_FILE):
        print("Data features.csv tidak ditemukan!")
        return

    df = pd.read_csv(INPUT_FILE)
    X = df.drop('label', axis=1)
    y = df['label']

    # Encode Label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print(f"Label Mapping: {label_mapping}")

    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 3. DEFINISI MODEL & GRID PARAMETER
    # Kita definisikan "Ruang Pencarian" (Search Space) untuk setiap model
    # Note: Kunci parameter harus diawali 'classifier__' karena masuk Pipeline
    
    if model_type == 'rf':
        base_clf = RandomForestClassifier(random_state=42)
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        
    elif model_type == 'svm':
        base_clf = SVC(probability=True, random_state=42)
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
        
    elif model_type == 'xgboost':
        base_clf = XGBClassifier(eval_metric='mlogloss', random_state=42)
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 6]
        }
        
    elif model_type == 'knn':
        base_clf = KNeighborsClassifier()
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }
        
    elif model_type == 'logreg':
        base_clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        param_grid = {
            'classifier__C': [0.1, 1, 10, 50]
        }
        
    else:
        print("Model tidak dikenal!")
        return

    # 4. MEMBUAT PIPELINE
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('classifier', base_clf)
    ])

    # 5. START MLFLOW RUN
    run_name = f"{model_type.upper()}_Tuned"
    
    with mlflow.start_run(run_name=run_name):
        print(f"Sedang mencari parameter terbaik (Tuning)...")
        
        # A. JALANKAN GRID SEARCH
        # cv=5 artinya 5-Fold Cross Validation (Sangat valid secara statistik)
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1, # Gunakan semua core CPU biar cepat
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Ambil Model Terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        
        print(f"\n[HASIL TUNING]")
        print(f"Parameter Terbaik: {best_params}")
        print(f"Akurasi Rata-rata Validasi (CV): {best_cv_score:.4f}")

        # B. LOG KE MLFLOW
        mlflow.log_param("model_type", model_type)
        # Log parameter terbaik (hapus prefix 'classifier__' biar bersih)
        clean_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
        mlflow.log_params(clean_params)
        mlflow.log_metric("best_cv_score", best_cv_score)
        
        # C. EVALUASI FINAL DI DATA TEST
        # (Data test ini BENAR-BENAR belum pernah dilihat saat tuning)
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        print(f"--> Akurasi Final (Data Test): {test_acc:.4f}")
        mlflow.log_metric("test_accuracy", test_acc)

        # D. Confusion Matrix Plot
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # Ganti warna jadi hijau biar beda
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'CM Tuned - {model_type.upper()} (Acc: {test_acc:.2f})')
        plt.ylabel('Label Asli')
        plt.xlabel('Prediksi Model')
        plt.tight_layout()
        
        cm_path = os.path.join(REPORT_PATH, f"cm_tuned_{model_type}.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # E. Simpan Model Terbaik
        signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:1]
        )
        
        print(f"Selesai! Model terbaik tersimpan di MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning Model Mood Classifier")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['rf', 'svm', 'xgboost', 'knn', 'logreg'],
                        help='Pilih algoritma untuk dituning')
    
    args = parser.parse_args()
    train(args.model)