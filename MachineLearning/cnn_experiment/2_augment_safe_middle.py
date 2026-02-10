import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil

# --- KONFIGURASI PATH ---
# Path Sumber (Lagu Full Asli)
INPUT_ROOT_TRAIN = 'data/split/train' 
INPUT_ROOT_TEST  = 'data/split/test'

# Path Tujuan (Folder Siap Training YAMNet)
OUTPUT_ROOT_TRAIN = 'data/interim_correct/train'
OUTPUT_ROOT_TEST  = 'data/interim_correct/test'

def trim_middle(y, sr, percentage=0.5):
    """
    Mengambil bagian tengah lagu sebanyak 'percentage'.
    Contoh: 0.5 berarti ambil 50% tengah (buang 25% awal, 25% akhir).
    """
    total_samples = len(y)
    keep_samples = int(total_samples * percentage)
    
    # Hitung start dan end
    start_idx = int((total_samples - keep_samples) / 2)
    end_idx = start_idx + keep_samples
    
    y_trimmed = y[start_idx:end_idx]
    
    # Safety: Kalau hasil potong kependekan (< 1 detik), kembalikan aslinya
    if len(y_trimmed) < sr:
        return y
        
    return y_trimmed

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_shift(y, sr, shift_max=2.0):
    shift = np.random.randint(sr * shift_max)
    return np.roll(y, shift)

def process_dataset(input_root, output_root, is_train=True):
    print(f"\n🚀 Memproses: {input_root} -> {output_root}")
    mode = "TRAINING (Augmented)" if is_train else "TESTING (Original Trimmed)"
    print(f"   Mode: {mode}")
    print("   Action: Trim Middle 50%" + (" + Noise + Shift" if is_train else ""))

    if not os.path.exists(input_root):
        print(f"❌ Error: Input {input_root} tidak ada.")
        return

    # Bersihkan output folder lama jika ada (biar fresh)
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    moods = os.listdir(input_root)
    
    for mood in moods:
        input_mood_path = os.path.join(input_root, mood)
        output_mood_path = os.path.join(output_root, mood)
        
        if not os.path.isdir(input_mood_path): continue
        os.makedirs(output_mood_path, exist_ok=True)
        
        files = [f for f in os.listdir(input_mood_path) if f.endswith(('.mp3', '.wav', '.flac'))]
        
        for f in tqdm(files, desc=f"Mood: {mood}"):
            try:
                # 1. Load Audio Full
                file_path = os.path.join(input_mood_path, f)
                y, sr = librosa.load(file_path, sr=22050)
                
                # 2. POTONG TENGAH (CRITICAL STEP)
                y = trim_middle(y, sr, percentage=0.5)
                
                # Simpan versi "Original" (yang sudah dipotong)
                # Gunakan format WAV agar cepat dibaca YAMNet nanti
                base_name = os.path.splitext(f)[0]
                sf.write(os.path.join(output_mood_path, base_name + '.wav'), y, sr)
                
                # 3. Lakukan Augmentasi (HANYA UNTUK DATA TRAIN)
                if is_train:
                    # Augmentasi Noise
                    y_noise = add_noise(y, noise_factor=0.005)
                    sf.write(os.path.join(output_mood_path, base_name + '_aug_noise.wav'), y_noise, sr)
                    
                    # Augmentasi Time Shift
                    y_shift = time_shift(y, sr, shift_max=2.0)
                    sf.write(os.path.join(output_mood_path, base_name + '_aug_shift.wav'), y_shift, sr)

            except Exception as e:
                print(f"⚠️ Gagal {f}: {e}")

def main():
    print("✂️ PEMROSESAN DATA: FOKUS TENGAH LAGU (50%)")
    
    # 1. Proses Data TRAIN (Trim + Augmentasi)
    process_dataset(INPUT_ROOT_TRAIN, OUTPUT_ROOT_TRAIN, is_train=True)
    
    # 2. Proses Data TEST (Trim Saja - Tanpa Augmentasi)
    # Penting: Data Test juga harus dipotong tengahnya agar konsisten evaluasinya
    process_dataset(INPUT_ROOT_TEST, OUTPUT_ROOT_TEST, is_train=False)
    
    print("\n✅ SELESAI SEMUA!")
    print(f"   Data siap di folder: data/interim_correct")
    print("   Sekarang jalankan '4_train_yamnet.py'.")

if __name__ == "__main__":
    main()