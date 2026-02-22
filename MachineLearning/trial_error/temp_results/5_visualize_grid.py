import os
import math
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- KONFIGURASI ---
INPUT_DIR = 'data/split_exp4/test' 
OUTPUT_DIR = 'debug_grids_middle_test' # Folder output baru biar gak ketukar
TARGET_SR = 16000
COLS = 5 

# --- FUNGSI POTONG TENGAH ---
def trim_middle(y):
    """Mengambil 50% bagian tengah audio"""
    length = len(y)
    # Jika audio kependekan (< 1 detik), jangan dipotong
    if length < 16000: 
        return y
        
    start = int(length * 0.25) # Buang 25% Awal
    end = int(length * 0.75)   # Buang 25% Akhir
    
    trimmed_y = y[start:end]
    
    # Safety: Kalau hasil potongannya rusak/kosong, kembalikan aslinya
    if len(trimmed_y) < 1600:
        return y
        
    return trimmed_y

def generate_mood_grid():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    print(f"🎨 Membuat Grid Spectrogram (50% Middle) dari {INPUT_DIR}...")
    
    for mood in moods:
        src_path = os.path.join(INPUT_DIR, mood)
        if not os.path.exists(src_path): continue
        
        files = sorted([f for f in os.listdir(src_path) if f.endswith(('.wav', '.mp3'))])
        num_files = len(files)
        
        if num_files == 0: continue

        print(f"\n📂 Processing {mood.upper()} ({num_files} lagu)...")
        
        rows = math.ceil(num_files / COLS)
        
        # Siapkan Kanvas Raksasa
        fig, axes = plt.subplots(rows, COLS, figsize=(20, 4 * rows), constrained_layout=True)
        
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = np.array([axes]).flatten()
            
        for i, ax in enumerate(tqdm(axes)):
            if i < num_files:
                f = files[i]
                file_path = os.path.join(src_path, f)
                
                try:
                    # 1. LOAD FULL AUDIO (Tanpa limit duration)
                    # Kita perlu full length untuk hitung posisi tengah yang akurat
                    y, sr = librosa.load(file_path, sr=TARGET_SR)
                    
                    # 2. POTONG TENGAH (Ambil 50% Inti)
                    y = trim_middle(y)
                    
                    # 3. Buat Spectrogram dari potongan tersebut
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    
                    # Plot
                    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', 
                                                   fmax=8000, ax=ax, cmap='magma')
                    
                    # Judul File
                    ax.set_title(f"{f[:20]}...", fontsize=9)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    
                except Exception as e:
                    ax.text(0.5, 0.5, "ERROR", ha='center', va='center')
                    print(f"Error {f}: {e}")
            else:
                ax.axis('off')
        
        save_path = os.path.join(OUTPUT_DIR, f"GRID_MIDDLE_{mood.upper()}.png")
        print(f"💾 Menyimpan gambar raksasa ke: {save_path}")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)

    print(f"\n✅ Selesai! Cek folder '{OUTPUT_DIR}'.")
    print("Gambar ini merepresentasikan apa yang 'dilihat' oleh Model Exp 7 Champion.")

if __name__ == "__main__":
    generate_mood_grid()