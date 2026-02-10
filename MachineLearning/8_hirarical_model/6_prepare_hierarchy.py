import os
import shutil
from tqdm import tqdm

# --- KONFIGURASI ---
# Gunakan dataset yang sudah di-augmentasi (dari Exp 4 atau Exp 7 Robust)
SOURCE_DIR = 'data/processed_exp4/train' 
BASE_OUT = 'data/hierarchy'

def setup_hierarchy():
    # Definisi 3 Dataset Khusus
    dirs = {
        'stage_1_arousal':    ['high', 'low'],             # Untuk Model 1 (High vs Low)
        'stage_2a_high_group': ['angry', 'happy'],         # Untuk Model 2A (Valence di High Arousal)
        'stage_2b_low_group':  ['sad', 'relaxed']          # Untuk Model 2B (Valence di Low Arousal)
    }
    
    # Reset Folder
    if os.path.exists(BASE_OUT): shutil.rmtree(BASE_OUT)
    
    for stage, classes in dirs.items():
        for c in classes:
            os.makedirs(os.path.join(BASE_OUT, stage, c), exist_ok=True)
            
    print("📂 Mengatur Ulang Dataset Hierarki (Thayer's Logic)...")
    
    # Mapping Logika Thayer
    mood_map = {
        # HIGH AROUSAL GROUP
        'angry':   {'s1': 'high', 's2': 'angry',   'group': 'stage_2a_high_group'},
        'happy':   {'s1': 'high', 's2': 'happy',   'group': 'stage_2a_high_group'},
        
        # LOW AROUSAL GROUP
        'sad':     {'s1': 'low',  's2': 'sad',     'group': 'stage_2b_low_group'},
        'relaxed': {'s1': 'low',  's2': 'relaxed', 'group': 'stage_2b_low_group'}
    }

    # Copy File
    for mood in mood_map.keys():
        src_mood_path = os.path.join(SOURCE_DIR, mood)
        if not os.path.exists(src_mood_path): continue
        
        files = os.listdir(src_mood_path)
        for f in tqdm(files, desc=f"Moving {mood}"):
            if not f.endswith('.wav'): continue
            src = os.path.join(src_mood_path, f)
            
            # 1. Copy ke Dataset Stage 1 (Label: High / Low)
            dst_s1 = os.path.join(BASE_OUT, 'stage_1_arousal', mood_map[mood]['s1'], f)
            shutil.copy(src, dst_s1)
            
            # 2. Copy ke Dataset Stage 2 (Label Spesifik)
            group_folder = mood_map[mood]['group']
            target_class = mood_map[mood]['s2']
            dst_s2 = os.path.join(BASE_OUT, group_folder, target_class, f)
            shutil.copy(src, dst_s2)

if __name__ == "__main__":
    setup_hierarchy()
    print("✅ Struktur Folder Siap! Data terbagi untuk 3 Model.")