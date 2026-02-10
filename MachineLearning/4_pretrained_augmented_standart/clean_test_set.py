import os
import shutil
import librosa
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- KONFIGURASI: SEKARANG KITA BERSIHKAN TEST ---
DATA_DIR = 'data/split_exp4/test' 
TARGET_SR = 16000

# Copy-Paste Daftar Target yang sama persis agar adil
TARGET_ACTIONS = {
    # --- HAPPY ---
    "Favorite Girl": "relaxed",
    "Daylight": "sad",
    "THE SHADE": "relaxed",
    "Blinding Lights": "DELETE", 
    "Thats So True": "sad",
    
    # --- SAD ---
    "505": "DELETE",
    "chamber of reflection": "DELETE",
    "Here With Me": "DELETE",
    "Love You Less": "DELETE",
    "rises the moon": "DELETE",
    "Good Looking": "DELETE",
    "Telephones": "DELETE",
    "Talking To The Moon": "relaxed", 
    "Thinking Out Loud": "relaxed",
    "Back To December": "relaxed",
    
    # --- ANGRY ---
    "Therefore I Am": "DELETE",
    "Gasoline": "DELETE",
    "mad woman": "DELETE",
    "10 Things I Hate About You": "DELETE",
    
    # --- RELAXED ---
    "I See Red": "angry", # PENTING!
    "Cinnamon Girl": "DELETE",
    "Radio": "DELETE",
    "Reckless": "DELETE",
    "Eldest Daughter": "DELETE",
    "Lose Control": "DELETE",
    "Romantic Homicide": "DELETE",
    "Get You": "DELETE",
    "Kingston": "DELETE",
    "Slow Dancing": "DELETE", 
    "her": "DELETE", 
    "THANK YOU 4 LOVIN": "DELETE",
    "wave to earth - bad": "DELETE"
}

def analyze_audio_thayer(file_path):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR, duration=30)
        rms = np.mean(librosa.feature.rms(y=y))
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        if isinstance(tempo, np.ndarray): tempo = tempo.item()
        return rms, tempo
    except: return 0, 0

def execute_test_cleaning():
    print("🧹 MEMBERSIHKAN DATA TEST (Menyamakan Standar)...")
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    for mood in moods:
        src_dir = os.path.join(DATA_DIR, mood)
        if not os.path.exists(src_dir): continue
        
        files = os.listdir(src_dir)
        print(f"\n📂 Checking {mood.upper()}...")
        
        for f in tqdm(files):
            full_path = os.path.join(src_dir, f)
            
            # 1. Cek Target List
            action = None
            for keyword, act in TARGET_ACTIONS.items():
                if keyword.lower() in f.lower():
                    action = act
                    break
            
            if action:
                if action == "DELETE":
                    os.remove(full_path)
                    print(f"  ❌ DELETE: {f}")
                elif action != mood:
                    dst_dir = os.path.join(DATA_DIR, action)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.move(full_path, os.path.join(dst_dir, f))
                    print(f"  ➡️ MOVE: {f} -> {action}")
                continue # Skip ke file berikutnya jika sudah kena action
            
            # 2. Cek Thayer (BPM & RMS)
            # Hanya jalankan jika file masih ada (belum dihapus/move di atas)
            if os.path.exists(full_path) and f.endswith('.wav'):
                rms, bpm = analyze_audio_thayer(full_path)
                reason = ""
                
                if mood in ['sad', 'relaxed'] and bpm > 125:
                    reason = f"BPM Tinggi ({bpm:.0f})"
                if mood == 'angry' and rms < 0.02:
                    reason = "Terlalu Pelan"
                
                if reason:
                    os.remove(full_path)
                    print(f"  ⚠️ AUTO-DELETE: {f} ({reason})")

if __name__ == "__main__":
    execute_test_cleaning()