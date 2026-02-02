import joblib
import pandas as pd
import os
import sys

# --- IMPORT DARI UTILS ---
# Kita wajib pakai fungsi ini agar cara hitung fitur sama persis dengan Training
try:
    from utils import extract_features_from_file
except ImportError:
    print("❌ Error Fatal: File 'utils.py' tidak ditemukan!")
    print("   Pastikan script ini berada di folder yang sama dengan utils.py")
    sys.exit()

# --- KONFIGURASI ---
# Ganti dengan lokasi file lagu yang ingin dites (Angry/Relaxed)
FILE_LAGU = r"C:\Users\andiz\Downloads\In The End [Official HD Music Video] - Linkin Park.mp3"
# Folder tempat menyimpan model (pastikan ada file .pkl di sini)
MODEL_DIR = 'models_final' 

def main():
    # 1. Load Aset Model
    print("⏳ Loading model dan aset...")
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'audio_mood_model.pkl'))
        cols = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl')) 
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        print("✅ Model berhasil diload.")
    except FileNotFoundError as e:
        print(f"❌ Gagal load model: {e}")
        print(f"   Pastikan folder '{MODEL_DIR}' berisi 3 file .pkl yang dibutuhkan.")
        return

    # 2. Cek File Lagu
    if not os.path.exists(FILE_LAGU):
        print(f"❌ File lagu tidak ditemukan: {FILE_LAGU}")
        return

    # 3. Ekstrak Fitur (Menggunakan utils.py)
    print(f"🎵 Menganalisa: {os.path.basename(FILE_LAGU)}...")
    
    # utils.py mengembalikan DICTIONARY, bukan Array
    features_dict = extract_features_from_file(FILE_LAGU)
    
    if features_dict is None:
        print("❌ Gagal mengekstrak fitur (File corrupt atau terlalu pendek).")
        return

    # 4. Format Data (PENTING!)
    # Ubah Dictionary ke DataFrame
    input_df = pd.DataFrame([features_dict])
    
    # Penyelarasan Kolom (Reindexing)
    # Ini menjamin urutan kolom 100% sama dengan saat training (75 kolom)
    # Jika ada kolom yang hilang/beda urutan, ini akan memperbaikinya otomatis.
    input_df = input_df.reindex(columns=cols, fill_value=0)

    # 5. Prediksi & Tampilkan Hasil
    print("\n" + "="*30)
    print("   HASIL DETEKSI MOOD")
    print("="*30)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        
        # Urutkan dari confidence tertinggi ke terendah
        indices = probs.argsort()[::-1] 
        
        for i in indices:
            mood_name = le.inverse_transform([i])[0]
            score = probs[i] * 100
            
            # Visualisasi Bar Chart Sederhana
            bar_length = int(score / 5) # 1 balok per 5%
            bar = "█" * bar_length
            
            # Print dengan format rapi
            print(f"{mood_name.upper().ljust(10)}: {score:.2f}%  {bar}")
            
    else:
        # Fallback jika model SVM dibuat tanpa probability=True
        pred_idx = model.predict(input_df)[0]
        mood_name = le.inverse_transform([pred_idx])[0]
        print(f"Mood Dominan: {mood_name.upper()}")

    print("="*30)

if __name__ == "__main__":
    main()