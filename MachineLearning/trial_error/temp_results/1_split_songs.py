import os
import shutil
import random

# --- KONFIGURASI ---
SOURCE_DIR = 'data/raw'
DEST_BASE = 'data/split' # Folder tujuan baru
TRAIN_RATIO = 0.8

def main():
    if os.path.exists(DEST_BASE):
        shutil.rmtree(DEST_BASE) # Hapus folder lama biar bersih
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    
    for mood in moods:
        src_mood_path = os.path.join(SOURCE_DIR, mood)
        if not os.path.exists(src_mood_path):
            print(f"⚠️ Folder mood tidak ditemukan: {mood}")
            continue
            
        # 1. Ambil semua file lagu
        files = [f for f in os.listdir(src_mood_path) if f.endswith(('.mp3', '.wav'))]
        random.shuffle(files) # Acak urutan lagu
        
        # 2. Hitung titik potong
        split_idx = int(len(files) * TRAIN_RATIO)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        print(f"📂 {mood.upper()}: Total {len(files)} | Train: {len(train_files)} | Test: {len(test_files)}")
        
        # 3. Pindahkan (Copy) ke Folder Baru
        for category, file_list in [('train', train_files), ('test', test_files)]:
            dest_dir = os.path.join(DEST_BASE, category, mood)
            os.makedirs(dest_dir, exist_ok=True)
            
            for f in file_list:
                shutil.copy(
                    os.path.join(src_mood_path, f),
                    os.path.join(dest_dir, f)
                )

    print("\n✅ PEMBAGIAN LAGU SELESAI!")
    print(f"📍 Data Train ada di: {os.path.join(DEST_BASE, 'train')}")
    print(f"📍 Data Test  ada di: {os.path.join(DEST_BASE, 'test')}")
    print("   (Data Test ini HARAM diintip oleh model saat training)")

if __name__ == "__main__":
    main()