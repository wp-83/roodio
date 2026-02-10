import os
import shutil
from tqdm import tqdm

# --- KONFIGURASI (WAJIB DIEDIT) ---
# Folder induk yang DI DALAMNYA ada folder "Q1", "Q2", "Q3", "Q4"
MERGE_ROOT_DIR = r'./data/merge_mir' 

# Output Folder (Data Training)
OUTPUT_DIR = 'data/training'

def prepare_merge_folders():
    print("📂 Menyiapkan Data MERGE (From Q1-Q4 Folders)...")
    
    if not os.path.exists(MERGE_ROOT_DIR):
        print(f"❌ Error: Folder {MERGE_ROOT_DIR} tidak ditemukan!")
        return

    # Mapping dari Folder Asli -> Folder Standar Kita
    # Pastikan nama folder di MERGE Anda sesuai (misal "Q1", "1", "Quadrant 1")
    # Di script ini saya asumsi namanya persis "Q1", "Q2", dll.
    mood_map = {
        'Q1': 'happy',
        'Q2': 'angry',
        'Q3': 'sad',
        'Q4': 'relaxed'
    }
    
    # Reset Output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    total_files = 0
    
    for source_folder, target_mood in mood_map.items():
        src_path = os.path.join(MERGE_ROOT_DIR, source_folder)
        dst_path = os.path.join(OUTPUT_DIR, target_mood)
        
        os.makedirs(dst_path, exist_ok=True)
        
        if not os.path.exists(src_path):
            print(f"⚠️ Warning: Folder sumber '{source_folder}' tidak ditemukan di {MERGE_ROOT_DIR}")
            continue
            
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.mp3', '.wav', '.flac'))]
        
        print(f"   ➡️ Mengcopy {source_folder} ke {target_mood} ({len(files)} files)...")
        
        for f in tqdm(files):
            try:
                # Tambahkan prefix 'merge_' biar rapi
                new_name = f"merge_{f}"
                shutil.copy2(os.path.join(src_path, f), os.path.join(dst_path, new_name))
                total_files += 1
            except Exception as e:
                print(f"Error copying {f}: {e}")

    print("\n" + "="*40)
    print("✅ DATASET MERGE SIAP!")
    print(f"🎯 Total File Terproses: {total_files}")
    print(f"📂 Lokasi: {os.path.abspath(OUTPUT_DIR)}")
    print("="*40)

if __name__ == "__main__":
    prepare_merge_folders()