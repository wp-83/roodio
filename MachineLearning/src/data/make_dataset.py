import os
import librosa
import soundfile as sf
import math

# Konfigurasi
RAW_DATA_PATH = 'data/raw'
OUTPUT_PATH = 'data/interim'
DURATION = 30  # Detik per potongan
SR = 22050     # Sample rate standar

def process_audio_files():
    # Loop ke setiap folder mood (happy, sad, etc)
    for mood in os.listdir(RAW_DATA_PATH):
        mood_path = os.path.join(RAW_DATA_PATH, mood)
        
        # Cek apakah itu folder
        if not os.path.isdir(mood_path):
            continue
            
        # Buat folder output jika belum ada
        output_mood_path = os.path.join(OUTPUT_PATH, mood)
        os.makedirs(output_mood_path, exist_ok=True)
        
        print(f"Memproses mood: {mood}...")
        
        # Loop ke setiap file lagu
        for filename in os.listdir(mood_path):
            file_path = os.path.join(mood_path, filename)
            
            try:
                # Load audio (bisa mp3 atau wav)
                y, sr = librosa.load(file_path, sr=SR)
                
                # Hitung durasi total
                total_duration = librosa.get_duration(y=y, sr=sr)
                
                # Hitung berapa banyak potongan yang bisa dibuat
                num_segments = math.floor(total_duration / DURATION)
                
                # Potong-potong
                for i in range(num_segments):
                    start_sample = i * DURATION * SR
                    end_sample = (i + 1) * DURATION * SR
                    
                    segment = y[start_sample:end_sample]
                    
                    # Simpan potongan ke folder interim
                    # Nama file jadi: laguasli_part1.wav, laguasli_part2.wav
                    new_filename = f"{os.path.splitext(filename)[0]}_part{i+1}.wav"
                    save_path = os.path.join(output_mood_path, new_filename)
                    
                    sf.write(save_path, segment, SR)
                    
            except Exception as e:
                print(f"Error memproses {filename}: {e}")

if __name__ == "__main__":
    process_audio_files()
    print("Selesai! Cek folder data/interim/")