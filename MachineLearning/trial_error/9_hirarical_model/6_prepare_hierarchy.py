import os
import shutil
from tqdm import tqdm

# --- KONFIGURASI ---
# Pastikan path ini BENAR mengarah ke folder yang ada isinya 'train' dan 'test'
SOURCE_DIR = 'data/processed_exp4' 
BASE_OUT = 'data/hierarchy'

def setup_hierarchy():
    # Hapus folder lama biar bersih
    if os.path.exists(BASE_OUT): shutil.rmtree(BASE_OUT)
    
    # Kita proses Train dan Test
    subsets = ['train', 'test']
    
    # Definisi Folder Tujuan
    stages = {
        'stage_1_arousal':     ['high', 'low'],
        'stage_2a_high_group': ['angry', 'happy'],
        'stage_2b_low_group':  ['sad', 'relaxed']
    }

    # Mapping Logika Thayer
    mood_map = {
        'angry':   {'s1': 'high', 's2': 'angry',   'group': 'stage_2a_high_group'},
        'happy':   {'s1': 'high', 's2': 'happy',   'group': 'stage_2a_high_group'},
        'sad':     {'s1': 'low',  's2': 'sad',     'group': 'stage_2b_low_group'},
        'relaxed': {'s1': 'low',  's2': 'relaxed', 'group': 'stage_2b_low_group'}
    }

    print("📂 Mengatur Ulang Dataset Hierarki...")

    for subset in subsets:
        print(f"\n🔄 Processing {subset.upper()} Set...")
        subset_path = os.path.join(SOURCE_DIR, subset)
        
        # Cek apakah source folder ada (misal data/processed_exp4/train)
        if not os.path.exists(subset_path):
            print(f"⚠️ Warning: Source folder tidak ditemukan: {subset_path}")
            continue

        # 1. Buat Struktur Folder
        for stage, classes in stages.items():
            for c in classes:
                os.makedirs(os.path.join(BASE_OUT, subset, stage, c), exist_ok=True)

        # 2. Pindahkan File
        for mood in mood_map.keys():
            src_mood_path = os.path.join(subset_path, mood)
            if not os.path.exists(src_mood_path): continue
            
            files = os.listdir(src_mood_path)
            for f in tqdm(files, desc=f"   Moving {mood}"):
                if not f.endswith(('.wav', '.mp3')): continue
                src = os.path.join(src_mood_path, f)
                
                # Copy ke Stage 1
                dst_s1 = os.path.join(BASE_OUT, subset, 'stage_1_arousal', mood_map[mood]['s1'], f)
                shutil.copy(src, dst_s1)
                
                # Copy ke Stage 2
                dst_s2 = os.path.join(BASE_OUT, subset, mood_map[mood]['group'], mood_map[mood]['s2'], f)
                shutil.copy(src, dst_s2)

if __name__ == "__main__":
    setup_hierarchy()
    print("\n✅ Struktur Folder Siap!")