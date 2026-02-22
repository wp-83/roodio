import os
import librosa
import soundfile as sf
import numpy as np

# HANYA Augmentasi Data TRAINING. Data Test JANGAN disentuh!
TRAIN_DIR = 'data/interim_correct/train'

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

def change_pitch(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def main():
    print("🚀 MEMULAI DATA AUGMENTATION (Hanya Train Set)...")
    
    if not os.path.exists(TRAIN_DIR):
        print("❌ Folder Train tidak ditemukan. Jalankan pemotong lagu dulu.")
        return

    for mood in os.listdir(TRAIN_DIR):
        mood_path = os.path.join(TRAIN_DIR, mood)
        if not os.path.isdir(mood_path): continue
        
        print(f"   ⚡ Mengaugmentasi genre: {mood}...")
        
        files = [f for f in os.listdir(mood_path) if f.endswith('.wav') and 'aug' not in f]
        
        for f in files:
            file_path = os.path.join(mood_path, f)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                
                # 1. Buat versi Noise
                y_noise = add_noise(y)
                sf.write(os.path.join(mood_path, f.replace('.wav', '_aug_noise.wav')), y_noise, sr)
                
                # 2. Buat versi Pitch Shift (Lebih rendah dikit biar variatif)
                y_pitch = change_pitch(y, sr, n_steps=-1.5) 
                sf.write(os.path.join(mood_path, f.replace('.wav', '_aug_pitch.wav')), y_pitch, sr)
                
            except Exception as e:
                print(f"Error {f}: {e}")

    print("\n✅ SELESAI! Data Training sekarang 3x lipat lebih banyak.")

if __name__ == "__main__":
    main()