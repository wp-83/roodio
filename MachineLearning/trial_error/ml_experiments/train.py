import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- KONFIGURASI PATH DINAMIS ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'features.csv'
FEEDBACK_PATH = BASE_DIR / 'new_feedback.csv' # Nanti dari Laravel taruh sini
MODELS_DIR = BASE_DIR / 'models'
MLRUNS_DIR = BASE_DIR / 'mlruns'

# Buat folder models jika belum ada
MODELS_DIR.mkdir(exist_ok=True)

def run_training():
    print("🚀 MEMULAI TRAINING PIPELINE...")

    # 1. SETUP MLFLOW
    # Menggunakan path absolut agar tidak error di Windows
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("Roodio_SVM_Retraining")

    # 2. LOAD DATASET UTAMA
    if not DATA_PATH.exists():
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan. Jalankan build_dataset.py dulu.")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"📚 Dataset Awal: {len(df)} sampel")

    # 3. CEK & MERGE DATA FEEDBACK (Retraining Logic)
    if FEEDBACK_PATH.exists():
        print("✨ Menemukan Data Feedback Baru!")
        try:
            df_feedback = pd.read_csv(FEEDBACK_PATH)
            
            # Validasi kolom harus sama
            if list(df.columns) == list(df_feedback.columns):
                df = pd.concat([df, df_feedback], ignore_index=True)
                print(f"📈 Total Data setelah digabung: {len(df)} sampel")
            else:
                print("⚠️ Struktur kolom feedback berbeda. Mengabaikan data baru.")
        except Exception as e:
            print(f"⚠️ Gagal membaca feedback: {e}")
    else:
        print("ℹ️ Tidak ada data feedback baru. Training dengan data awal saja.")

    # 4. PREPROCESSING
    X = df.drop('label', axis=1)
    y = df['label']

    # Simpan Urutan Kolom (PENTING untuk API)
    joblib.dump(list(X.columns), MODELS_DIR / 'feature_columns.pkl')

    # Encode Label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Simpan Encoder (PENTING untuk API)
    joblib.dump(le, MODELS_DIR / 'label_encoder.pkl')
    
    print(f"🏷️ Classes: {le.classes_}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 5. DEFINISI MODEL (SVM + SCALER)
    # SVM butuh scaling data
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True)) # Probability=True biar ada confidence score
    ])

    # 6. TRAINING DENGAN MLFLOW
    with mlflow.start_run() as run:
        print("🧠 Sedang melatih model SVM...")
        svm_pipeline.fit(X_train, y_train)

        # Evaluasi
        preds = svm_pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print("\n" + "="*30)
        print(f"🏆 AKURASI MODEL: {acc:.2%}")
        print("="*30)

        # Log Metrics & Parameters ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("total_samples", len(df))

        # Log Model Fisik ke dalam MLruns (Artifacts)
        mlflow.sklearn.log_model(svm_pipeline, "model")

        # Simpan Model Fisik ke Folder Lokal (Untuk dicopy ke Deploy)
        model_path = MODELS_DIR / 'audio_mood_model.pkl'
        joblib.dump(svm_pipeline, model_path)
        print(f"💾 Model tersimpan siap deploy: {model_path}")
        print(f"📂 Encoder & Columns tersimpan di: {MODELS_DIR}")

if __name__ == "__main__":
    run_training()