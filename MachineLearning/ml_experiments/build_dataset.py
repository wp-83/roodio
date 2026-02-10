import os
import pandas as pd
# Kita panggil otak yang sama dengan Server agar konsisten 100%
from utils import extract_features_from_file 

# --- KONFIGURASI ---
# Ubah input ke 'data/interim' sesuai workflow Anda (file potongan 30 detik)
INPUT_DATA_DIR = 'data/interim'        
OUTPUT_CSV = 'data/processed/features.csv'

def main():
    if not os.path.exists(INPUT_DATA_DIR):
        print(f"❌ Folder {INPUT_DATA_DIR} tidak ditemukan!")
        print("⚠️ Pastikan Anda sudah menjalankan script pemotong lagu (make_dataset) sebelumnya.")
        return

    data = []
    
    # Cek apakah folder interim kosong
    if not os.listdir(INPUT_DATA_DIR):
        print(f"❌ Folder {INPUT_DATA_DIR} kosong.")
        return

    # Loop Folder Mood (Labels)
    for label in os.listdir(INPUT_DATA_DIR):
        mood_dir = os.path.join(INPUT_DATA_DIR, label)
        
        if not os.path.isdir(mood_dir):
            continue
            
        print(f"📂 Memproses Folder Interim: {label}...")
        
        files = os.listdir(mood_dir)
        count = 0
        
        # Loop File Audio Potongan
        for filename in files:
            file_path = os.path.join(mood_dir, filename)
            
            # Hanya proses file audio
            if not filename.lower().endswith(('.mp3', '.wav', '.ogg')):
                continue

            # --- PENTING: MENGGUNAKAN UTILS.PY ---
            # Kita pakai fungsi dari utils.py agar cara hitung MFCC, ZCR, dll
            # SAMA PERSIS dengan yang akan dilakukan Server Hugging Face nanti.
            features = extract_features_from_file(file_path)
            
            if features:
                features['label'] = label # Tambahkan label
                data.append(features)
                count += 1
        
        print(f"   --> Berhasil mengekstrak {count} sampel.")

    # Simpan ke CSV
    if data:
        df = pd.DataFrame(data)
        
        # Pindah label ke kolom terakhir biar rapi
        cols = [c for c in df.columns if c != 'label'] + ['label']
        df = df[cols]
        
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        
        print("\n" + "="*50)
        print(f"✅ DATASET FINAL SELESAI DIBUAT!")
        print(f"📍 Lokasi: {OUTPUT_CSV}")
        print(f"📊 Total Data Training: {len(df)} sampel")
        print("="*50)
    else:
        print("❌ Tidak ada data audio yang berhasil diekstrak.")

if __name__ == "__main__":
    main()