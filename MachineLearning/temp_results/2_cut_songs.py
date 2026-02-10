import os
import librosa
import soundfile as sf
import math

# --- KONFIGURASI BARU ---
# Mengambil dari folder hasil split langkah sebelumnya
INPUT_ROOT = 'data/split'       
# Menyimpan ke folder sementara baru (biar gak kecampur data lama)
OUTPUT_ROOT = 'data/interim_correct' 

DURATION = 30  # Detik per potongan
SR = 22050     # Sample rate standar

def process_subset(subset_name):
    """
    Memproses folder 'train' atau 'test' secara terpisah
    """
    input_base_path = os.path.join(INPUT_ROOT, subset_name)
    output_base_path = os.path.join(OUTPUT_ROOT, subset_name)
    
    print(f"\n🚀 Memproses Dataset: {subset_name.upper()}...")

    if not os.path.exists(input_base_path):
        print(f"❌ Folder tidak ditemukan: {input_base_path}")
        return

    # Loop ke setiap folder mood (happy, sad, etc)
    for mood in os.listdir(input_base_path):
        mood_path = os.path.join(input_base_path, mood)
        
        # Cek apakah itu folder
        if not os.path.isdir(mood_path):
            continue
            
        # Buat folder output (contoh: data/interim_correct/train/happy)
        output_mood_path = os.path.join(output_base_path, mood)
        os.makedirs(output_mood_path, exist_ok=True)
        
        print(f"   📂 Genre: {mood}...")
        
        # Loop ke setiap file lagu
        for filename in os.listdir(mood_path):
            file_path = os.path.join(mood_path, filename)
            
            # Skip jika bukan file audio
            if not filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
                continue

            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=SR)
                
                # Hitung durasi total
                total_duration = librosa.get_duration(y=y, sr=sr)
                
                # Hitung berapa banyak potongan yang bisa dibuat
                num_segments = math.floor(total_duration / DURATION)
                
                if num_segments == 0:
                    print(f"      ⚠️ Skip {filename}: Terlalu pendek (< {DURATION}s)")
                    continue

                # Potong-potong
                for i in range(num_segments):
                    start_sample = i * DURATION * SR
                    end_sample = (i + 1) * DURATION * SR
                    
                    segment = y[start_sample:end_sample]
                    
                    # Simpan potongan
                    # Format: namalagu_part1.wav
                    new_filename = f"{os.path.splitext(filename)[0]}_part{i+1}.wav"
                    save_path = os.path.join(output_mood_path, new_filename)
                    
                    sf.write(save_path, segment, SR)
                    
            except Exception as e:
                print(f"      ❌ Error memproses {filename}: {e}")

def main():
    # Proses Training Data
    process_subset('train')
    
    # Proses Testing Data
    process_subset('test')
    
    print("\n" + "="*50)
    print("✅ PEMOTONGAN SELESAI!")
    print(f"📍 Hasil potongan ada di folder: {OUTPUT_ROOT}")
    print("="*50)

if __name__ == "__main__":
    main()