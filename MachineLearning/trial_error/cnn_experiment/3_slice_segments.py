import os
import librosa
import numpy as np
from tqdm import tqdm

# --- KONFIGURASI ---
# Menggunakan '../' karena script ada di dalam folder cnn_experiment
# Kita ambil dari split_raw (Lagu Full Train/Test)
INPUT_ROOT = './data/interim_correct' 
OUTPUT_ROOT = './data/processed_cnn'  # Output Folder

SEGMENT_LEN = 3 # Durasi potongan (detik)
SR = 22050      # Sample Rate

def process_data():
    print("🔪 Memotong LAGU FULL jadi segmen 3 detik & Konversi ke Spektrogram...")
    print(f"   Sumber Data: {os.path.abspath(INPUT_ROOT)}")
    print(f"   Tujuan Data: {os.path.abspath(OUTPUT_ROOT)}")
    
    if not os.path.exists(INPUT_ROOT):
        print("❌ Error: Folder Input tidak ditemukan!")
        print("   Pastikan Anda sudah menjalankan '1_split_songs.py' sebelumnya.")
        return

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # Loop Train & Test
    for subset in ['train', 'test']:
        input_dir = os.path.join(INPUT_ROOT, subset)
        output_dir = os.path.join(OUTPUT_ROOT, subset)
        
        if not os.path.exists(input_dir):
            print(f"⚠️ Warning: Folder {subset} tidak ada di {INPUT_ROOT}")
            continue
        
        # Loop Mood (Happy, Sad, etc)
        for mood in os.listdir(input_dir):
            mood_in = os.path.join(input_dir, mood)
            mood_out = os.path.join(output_dir, mood)
            
            if not os.path.isdir(mood_in): continue
            
            os.makedirs(mood_out, exist_ok=True)
            
            print(f"   📂 Processing {subset}/{mood}...")
            
            files = [f for f in os.listdir(mood_in) if f.endswith(('.mp3', '.wav', '.flac'))]
            
            for f in tqdm(files):
                file_path = os.path.join(mood_in, f)
                try:
                    # 1. Load Audio Full (Bisa 3-4 menit)
                    # Hati-hati: Load lagu full makan RAM. Kalau error Memory, 
                    # kita perlu load per bagian (streaming), tapi biasanya 3 menit masih aman.
                    y, _ = librosa.load(file_path, sr=SR)
                    
                    # 2. Hitung jumlah sampel per 3 detik
                    samples_per_segment = SEGMENT_LEN * SR
                    total_segments = len(y) // samples_per_segment
                    
                    # Jika lagu terlalu pendek (< 3 detik), skip
                    if total_segments == 0: continue

                    # 3. Potong & Buat Spektrogram
                    for i in range(total_segments):
                        start = i * samples_per_segment
                        end = start + samples_per_segment
                        segment = y[start:end]
                        
                        # --- FITUR VISUAL UNTUK CNN ---
                        # Buat Mel-Spectrogram (GAMBAR SUARA)
                        # n_mels=128 = Tinggi Gambar 128 pixel
                        melspec = librosa.feature.melspectrogram(y=segment, sr=SR, n_mels=128)
                        melspec_db = librosa.power_to_db(melspec, ref=np.max)
                        
                        # Normalisasi ke 0-1 (Agar CNN mudah belajar)
                        # Rumus Min-Max Scaling
                        min_val = melspec_db.min()
                        max_val = melspec_db.max()
                        if max_val - min_val == 0:
                            melspec_norm = np.zeros_like(melspec_db)
                        else:
                            melspec_norm = (melspec_db - min_val) / (max_val - min_val)
                        
                        # Simpan sebagai file numpy (.npy)
                        # Hemat tempat & loading lebih cepat daripada .png/.jpg
                        out_name = f"{os.path.splitext(f)[0]}_seg{i}.npy"
                        save_path = os.path.join(mood_out, out_name)
                        np.save(save_path, melspec_norm)
                        
                except Exception as e:
                    print(f"❌ Error {f}: {e}")

if __name__ == "__main__":
    process_data()