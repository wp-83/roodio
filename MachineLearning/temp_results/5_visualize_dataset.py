import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- KONFIGURASI ---
# Gunakan folder data yang ingin dicek (misal split_exp4 atau raw)
INPUT_DIR = 'data/split_exp4/train' 
OUTPUT_DIR = 'debug_spectrograms'
TARGET_SR = 16000

def create_spectrogram(audio_path, save_path, title):
    try:
        # Load 30 detik saja biar cepat
        y, sr = librosa.load(audio_path, sr=TARGET_SR, duration=30)
        
        # Buat Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plotting (Tanpa Axis biar fokus ke pola)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        # Simpan
        plt.savefig(save_path)
        plt.close() # Penting biar memori gak bocor
        
    except Exception as e:
        print(f"Error {title}: {e}")

def generate_gallery():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    print(f"🎨 Membuat Galeri Spectrogram dari {INPUT_DIR}...")
    
    for mood in moods:
        src_path = os.path.join(INPUT_DIR, mood)
        dst_path = os.path.join(OUTPUT_DIR, mood)
        os.makedirs(dst_path, exist_ok=True)
        
        if not os.path.exists(src_path): continue
        
        files = [f for f in os.listdir(src_path) if f.endswith(('.wav', '.mp3'))]
        
        # Sortir biar rapi
        files.sort()
        
        print(f"📂 Processing {mood.upper()}...")
        for f in tqdm(files):
            file_path = os.path.join(src_path, f)
            save_name = os.path.splitext(f)[0] + ".png"
            save_path = os.path.join(dst_path, save_name)
            
            create_spectrogram(file_path, save_path, f"{mood} - {f}")

    print(f"\n✅ Selesai! Cek folder '{OUTPUT_DIR}' sekarang.")

if __name__ == "__main__":
    generate_gallery()