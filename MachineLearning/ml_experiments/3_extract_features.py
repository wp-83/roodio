import os
import pandas as pd
from utils import extract_features_from_file # Import utils Anda

# --- KONFIGURASI ---
INPUT_ROOT = 'data/interim_correct'
OUTPUT_CSV_DIR = 'data/processed_correct'

def create_csv(subset_name):
    print(f"\n☕ Mengekstrak Fitur: {subset_name.upper()} (Sabar ya, ini lama)...")
    
    input_base = os.path.join(INPUT_ROOT, subset_name)
    data = []
    
    if not os.path.exists(input_base):
        print("❌ Folder tidak ditemukan.")
        return

    # Loop Mood
    for label in os.listdir(input_base):
        mood_dir = os.path.join(input_base, label)
        if not os.path.isdir(mood_dir): continue
        
        print(f"   Genre: {label}...")
        
        # Loop File Wav
        for filename in os.listdir(mood_dir):
            file_path = os.path.join(mood_dir, filename)
            
            # Panggil Utils
            features = extract_features_from_file(file_path)
            
            if features:
                features['label'] = label
                data.append(features)
    
    # Simpan CSV
    if data:
        df = pd.DataFrame(data)
        # Rapikan kolom
        cols = [c for c in df.columns if c != 'label'] + ['label']
        df = df[cols]
        
        os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_CSV_DIR, f'{subset_name}_features.csv')
        df.to_csv(out_path, index=False)
        print(f"✅ Selesai! Disimpan di: {out_path}")
    else:
        print("❌ Data kosong.")

if __name__ == "__main__":
    create_csv('train')
    create_csv('test')