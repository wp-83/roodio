import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil

# --- KONFIGURASI PATH (Sesuai Permintaan) ---
# 1. Path Sumber (Lagu Asli/Full Song)
INPUT_ROOT = 'data/split/train' 

# 2. Path Tujuan (Folder Master Training: Asli + Augmentasi)
OUTPUT_ROOT = 'data/interim_correct/train'

TEST_INPUT = 'data/split/test'
TEST_OUTPUT = 'data/interim_correct/test'

def add_noise(y, noise_factor=0.005):
    """Menambahkan random noise (kresek halus)."""
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

def time_shift(y, sr, shift_max=2.0):
    """Menggeser audio secara horizontal (rolling)."""
    shift = np.random.randint(sr * shift_max)
    return np.roll(y, shift)

def process_augmentation():
    print("🛡️ MEMULAI SAFE AUGMENTATION (From Split -> Interim Correct)...")
    print(f"   📂 Sumber Data: {INPUT_ROOT}")
    print(f"   📂 Simpan ke  : {OUTPUT_ROOT}")
    print("   Strategi: Copy Original + Noise Injection + Time Shift")
    
    # Cek folder input
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Error: Folder Input {INPUT_ROOT} tidak ditemukan.")
        print("   Pastikan Anda sudah menjalankan '1_split_songs.py'.")
        return

    # Hapus folder output lama jika ada (Biar bersih dari sisa eksperimen gagal)
    # Hati-hati: Aktifkan baris ini jika ingin folder benar-benar fresh
    # if os.path.exists(OUTPUT_ROOT): shutil.rmtree(OUTPUT_ROOT)

    # Buat folder output
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    moods = os.listdir(INPUT_ROOT)
    
    for mood in moods:
        input_mood_path = os.path.join(INPUT_ROOT, mood)
        output_mood_path = os.path.join(OUTPUT_ROOT, mood)
        
        # Skip jika bukan folder
        if not os.path.isdir(input_mood_path): continue
        
        # Buat sub-folder mood di output (misal: interim_correct/train/happy)
        os.makedirs(output_mood_path, exist_ok=True)
        
        print(f"\n⚡ Memproses Genre: {mood.upper()}...")
        
        # Ambil file asli saja dari source
        files = [f for f in os.listdir(input_mood_path) if f.endswith(('.mp3', '.wav', '.flac'))]
        
        for f in tqdm(files):
            input_file_path = os.path.join(input_mood_path, f)
            
            try:
                # Load Audio
                y, sr = librosa.load(input_file_path, sr=22050)
                
                # --- 1. SIMPAN ORIGINAL (Copy) ---
                sf.write(os.path.join(output_mood_path, f.replace('.mp3', '.wav')), y, sr)
                
                # --- 2. AUGMENTASI NOISE ---
                y_noise = add_noise(y, noise_factor=0.005)
                sf.write(os.path.join(output_mood_path, f.replace('.mp3', '').replace('.wav', '') + '_aug_noise.wav'), y_noise, sr)
                
                # --- 3. AUGMENTASI TIME SHIFT ---
                y_shift = time_shift(y, sr, shift_max=2.0)
                sf.write(os.path.join(output_mood_path, f.replace('.mp3', '').replace('.wav', '') + '_aug_shift.wav'), y_shift, sr)

            except Exception as e:
                print(f"⚠️ Gagal memproses {f}: {e}")
   
    print("\n🚚 Menyalin Data Test (Tanpa Augmentasi)...")
    if os.path.exists(TEST_INPUT):
        if os.path.exists(TEST_OUTPUT): shutil.rmtree(TEST_OUTPUT)
        shutil.copytree(TEST_INPUT, TEST_OUTPUT)
        print("✅ Data Test berhasil disalin.")
    else:
        print("⚠️ Folder Test source tidak ditemukan.")

    print("\n✅ SELESAI!")
    print(f"   Data Master Training siap di: {OUTPUT_ROOT}")
    print("   Sekarang arahkan script '4_train_yamnet.py' ke folder ini!")

if __name__ == "__main__":
    process_augmentation()