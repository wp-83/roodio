import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# --- KONFIGURASI ---
# Pastikan path ini sesuai dengan output script extract sebelumnya
TRAIN_CSV = 'data/processed_correct/train_features.csv'
TEST_CSV  = 'data/processed_correct/test_features.csv'

# Kita simpan di folder baru biar model lama tidak tertimpa dulu
MODEL_DIR = 'models_final' 

def main():
    print("🔥 MEMULAI TRAINING (METODE JUJUR/ANTI-LEAKAGE)...")
    
    # 1. Cek File
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print("❌ Error: File CSV tidak ditemukan.")
        print("   Pastikan Anda sudah menjalankan script '3_extract_features.py' sebelumnya.")
        return

    # 2. Load Data
    print("⏳ Membaca data...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    
    print(f"   📊 Data Training : {len(df_train)} sampel")
    print(f"   📊 Data Testing  : {len(df_test)} sampel (Lagu Asing)")
    
    # 3. Pisahkan Fitur (X) dan Label (y)
    # Hapus kolom 'label' dan 'filename' (jika ada)
    drop_cols = ['label']
    if 'filename' in df_train.columns: drop_cols.append('filename')

    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    y_train_raw = df_train['label']
    
    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    y_test_raw = df_test['label']
    
    # 4. Encoding Label (Ubah 'happy' jadi 0, 1, dst)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    # PENTING: Data test harus di-transform pakai encoder milik data train
    y_test  = le.transform(y_test_raw) 
    
    # 5. Definisikan Pipeline Model
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Normalisasi angka agar setara
        ('svm', SVC(
            kernel='rbf', 
            C=10,             # C lebih besar = lebih ketat
            gamma='scale',
            probability=True, # Wajib True agar bisa keluar persentase %
            class_weight='balanced', # <-- SOLUSI ANGRY/RELAXED (Menyeimbangkan kelas)
            random_state=42
        ))
    ])
    
    # 6. Proses Training
    print("🏋️ Sedang melatih model (Mungkin butuh waktu)...")
    pipeline.fit(X_train, y_train)
    print("✅ Training selesai!")
    
    # 7. Evaluasi Jujur (Real World Test)
    print("\n" + "="*50)
    print("   HASIL EVALUASI PADA DATA TESTING (UNSEEN DATA)")
    print("="*50)
    
    y_pred = pipeline.predict(X_test)
    
    # Tampilkan Report Angka
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 8. Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix (Akurasi Murni)')
    plt.xlabel('Prediksi AI')
    plt.ylabel('Kunci Jawaban Asli')
    plt.tight_layout()
    plt.show() # Akan memunculkan jendela grafik

    # 9. Simpan Model Akhir
    print(f"\n💾 Menyimpan model ke folder '{MODEL_DIR}'...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'audio_mood_model.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    
    # Simpan nama kolom fitur (Penting untuk app.py nanti)
    feature_columns = list(X_train.columns)
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    
    print("🎉 SELESAI! Model siap digunakan.")

if __name__ == "__main__":
    main()