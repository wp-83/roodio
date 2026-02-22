import os
import shutil
import librosa
import numpy as np
import warnings
from tqdm import tqdm

# Matikan warning
warnings.filterwarnings("ignore")

# --- KONFIGURASI ---
DATA_DIR = 'data/split_exp4/train' # Fokus bersihkan Train dulu
TARGET_SR = 16000

# ==========================================
# TAHAP 1: DAFTAR TARGET (Berdasarkan Analisis Data Curator)
# Format: "Kata Kunci File": "Aksi"
# Aksi: 'DELETE' atau nama folder tujuan (misal 'relaxed', 'angry')
# ==========================================
TARGET_ACTIONS = {
    # --- HAPPY ---
    "Favorite Girl": "relaxed",
    "Daylight": "sad",
    "THE SHADE": "relaxed",
    "Blinding Lights": "DELETE", # Quality issue
    "Thats So True": "sad",
    
    # --- SAD (Buang lagu > 120 BPM) ---
    "505": "DELETE",
    "chamber of reflection": "DELETE",
    "Here With Me": "DELETE",
    "Love You Less": "DELETE",
    "rises the moon": "DELETE",
    "Good Looking": "DELETE",
    "Telephones": "DELETE",
    "Talking To The Moon": "relaxed", # Piano ballad, masuk relaxed
    "Thinking Out Loud": "relaxed",
    "Back To December": "relaxed",
    
    # --- ANGRY (Buang lagu pelan) ---
    "Therefore I Am": "DELETE",
    "Gasoline": "DELETE",
    "mad woman": "DELETE",
    "10 Things I Hate About You": "DELETE",
    
    # --- RELAXED (Buang lagu ngebut/berisik) ---
    "I See Red": "angry", # INI PENTING! Pindah ke Angry
    "Cinnamon Girl": "DELETE",
    "Radio": "DELETE",
    "Reckless": "DELETE",
    "Eldest Daughter": "DELETE",
    "Lose Control": "DELETE",
    "Romantic Homicide": "DELETE",
    "Get You": "DELETE",
    "Kingston": "DELETE",
    "Slow Dancing": "DELETE", # Joji ini terlalu dinamis
    "her": "DELETE", # JVKE
    "THANK YOU 4 LOVIN": "DELETE",
    "wave to earth - bad": "DELETE"
}

def analyze_audio_thayer(file_path):
    """Cek BPM dan RMS untuk Safety Net"""
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR, duration=30)
        rms = np.mean(librosa.feature.rms(y=y))
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        if isinstance(tempo, np.ndarray): tempo = tempo.item()
        return rms, tempo
    except:
        return 0, 0

def execute_cleaning():
    print("🧹 MULAI PEMBERSIHAN DATASET (THAYER MODEL)...")
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    processed_files = set() # Agar tidak diproses 2x
    
    # --- STEP 1: EKSEKUSI DAFTAR TARGET ---
    print("\n[STEP 1] Menjalankan Daftar Target Manual...")
    for mood in moods:
        src_dir = os.path.join(DATA_DIR, mood)
        if not os.path.exists(src_dir): continue
        
        files = os.listdir(src_dir)
        for f in files:
            full_path = os.path.join(src_dir, f)
            
            # Cek apakah file ini ada di daftar target?
            action = None
            for keyword, act in TARGET_ACTIONS.items():
                if keyword.lower() in f.lower():
                    action = act
                    break
            
            if action:
                processed_files.add(full_path)
                if action == "DELETE":
                    os.remove(full_path)
                    print(f"❌ DELETE: {f} (Dari {mood})")
                elif action != mood: # Pindah Folder
                    dst_dir = os.path.join(DATA_DIR, action)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.move(full_path, os.path.join(dst_dir, f))
                    print(f"➡️ MOVE: {f} | {mood.upper()} -> {action.upper()}")

    # --- STEP 2: SAFETY NET (Cek BPM & RMS Otomatis) ---
    print("\n[STEP 2] Scanning Sisa File dengan Algoritma Thayer...")
    
    for mood in moods:
        src_dir = os.path.join(DATA_DIR, mood)
        files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]
        
        for f in tqdm(files, desc=f"Scanning {mood}"):
            full_path = os.path.join(src_dir, f)
            if full_path in processed_files: continue # Skip yang sudah diproses
            
            rms, bpm = analyze_audio_thayer(full_path)
            
            # ATURAN THAYER (SAFETY NET)
            reason = ""
            
            # 1. SAD & RELAXED: Haram BPM Tinggi (>125)
            if mood in ['sad', 'relaxed']:
                if bpm > 125:
                    reason = f"BPM Tinggi ({bpm:.0f})"
            
            # 2. ANGRY: Haram RMS Rendah (Terlalu pelan < 0.05)
            if mood == 'angry':
                if rms < 0.02: # Ambang batas hening
                    reason = "Terlalu Pelan/Hening"
            
            # EKSEKUSI
            if reason:
                # Kita pindahkan ke folder 'quarantine' dulu biar aman, atau delete
                # Di sini saya set DELETE agar tegas sesuai permintaan
                os.remove(full_path)
                print(f"⚠️ AUTO-DELETE ({mood}): {f} | Alasan: {reason}")

    print("\n✅ PEMBERSIHAN SELESAI!")
    print("Dataset Anda sekarang High Quality & Low Variance.")
    print("👉 Silakan jalankan: 2_prepare... -> 3_train...")

if __name__ == "__main__":
    execute_cleaning()