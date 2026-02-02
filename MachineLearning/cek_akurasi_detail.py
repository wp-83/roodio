import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os

# --- KONFIGURASI ---
DATA_PATH = 'data/processed/features.csv'
MODEL_DIR = 'models' 

def evaluasi_model():
    print("🚀 Memulai Evaluasi Detail Model...")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan.")
        return
    
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"❌ Gagal membaca CSV: {e}")
        return

    # 2. Pisahkan Fitur (X) dan Label (y)
    if 'filename' in df.columns:
        X = df.drop(columns=['label', 'filename'])
    else:
        X = df.drop(columns=['label'])
        
    y = df['label'] # <-- Ini isinya sudah teks ('happy', 'sad', dll)

    # 3. Load Model
    print("⏳ Loading model...")
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'audio_mood_model.pkl'))
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        print("✅ Model diload.")
    except:
        print("❌ Gagal load model. Pastikan folder 'models' benar.")
        return

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Prediksi
    print(f"🔍 Menguji {len(X_test)} sampel...")
    y_pred = model.predict(X_test) # <-- Ini hasilnya Angka (0, 1, 2...)
    
    # --- PERBAIKAN DI SINI ---
    
    # y_test TIDAK PERLU di-convert karena sudah teks
    y_test_labels = y_test 
    
    # y_pred PERLU di-convert dari Angka ke Teks
    y_pred_labels = le.inverse_transform(y_pred)
    
    # Ambil daftar kelas untuk label grafik
    classes = le.classes_

    # 6. Laporan
    print("\n" + "="*40)
    print("   LAPORAN AKURASI")
    print("="*40)
    print(classification_report(y_test_labels, y_pred_labels))

    # 7. Gambar Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix: Dimana AI Bingung?')
    plt.ylabel('Kunci Jawaban (Asli)')
    plt.xlabel('Tebakan AI')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluasi_model()