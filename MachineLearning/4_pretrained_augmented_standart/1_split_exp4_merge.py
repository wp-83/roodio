import os
import shutil
import random
from pathlib import Path

# --- 1. KUNCI RANDOM (SEED) ---
# Ini menjamin file train dan test SELALU SAMA setiap kali di-run
def set_seed(seed=34):
    random.seed(seed)
    print(f"🔒 Random Seed dikunci: {seed}")

set_seed(34) # Panggil di paling atas!

# --- KONFIGURASI ---
SOURCES = {
    'old': 'data/raw',   # Data Lama
    'new': 'data/raw2'   # Data Baru
}
OUTPUT_SPLIT_DIR = 'data/split_exp4'

# Rasio Split (0.8 artinya 80% masuk Train)
TRAIN_RATIO = 0.8

def split_merge_data():
    print("⚖️ MERGING & SPLITTING (RAW + RAW2) [FIXED SEED]...")
    
    if os.path.exists(OUTPUT_SPLIT_DIR):
        shutil.rmtree(OUTPUT_SPLIT_DIR)
    
    moods = ['happy', 'sad', 'angry', 'relaxed']
    total_train = 0
    total_test = 0
    
    for mood in moods:
        # 1. Kumpulkan semua file dari kedua folder
        all_candidates = []
        
        # Urutkan source key agar urutan pembacaan selalu sama (old dulu baru new, atau sebaliknya)
        for source_name in sorted(SOURCES.keys()): 
            source_path = SOURCES[source_name]
            mood_path = os.path.join(source_path, mood)
            
            if not os.path.exists(mood_path):
                print(f"⚠️ Warning: Folder {mood_path} tidak ditemukan.")
                continue
            
            # PENTING: Sortir nama file sebelum di-append!
            # os.listdir() tidak menjamin urutan, jadi wajib di-sort.
            files = sorted([f for f in os.listdir(mood_path) if f.endswith(('.mp3', '.wav'))])
            
            for f in files:
                # Simpan tuple: (nama_file_asli, path_lengkap, asal_sumber)
                full_path = os.path.join(mood_path, f)
                all_candidates.append((f, full_path, source_name))
        
        # 2. Acak Total (Terkunci Seed)
        # Sortir lagi list final berdasarkan path sebelum di-shuffle untuk kepastian total
        all_candidates.sort(key=lambda x: x[1]) 
        
        # Sekarang baru di-shuffle (hasilnya pasti sama terus)
        random.shuffle(all_candidates)
        
        # 3. Hitung Split
        n_total = len(all_candidates)
        n_train = int(n_total * TRAIN_RATIO)
        
        train_set = all_candidates[:n_train]
        test_set = all_candidates[n_train:]
        
        # 4. Copy ke Tujuan
        for subset_name, dataset in [('train', train_set), ('test', test_set)]:
            target_dir = os.path.join(OUTPUT_SPLIT_DIR, subset_name, mood)
            os.makedirs(target_dir, exist_ok=True)
            
            for fname, fpath, source in dataset:
                # Rename biar unik & ketahuan asalnya: "new_judul_lagu.wav"
                new_name = f"{source}_{fname}"
                shutil.copy2(fpath, os.path.join(target_dir, new_name))
        
        total_train += len(train_set)
        total_test += len(test_set)
        print(f"   📂 {mood.upper().ljust(8)} | Total: {n_total} -> Train: {len(train_set)} | Test: {len(test_set)}")

    print(f"\n✅ Total Gabungan: {total_train + total_test} File")
    print(f"   - Train: {total_train}")
    print(f"   - Test : {total_test}")

if __name__ == "__main__":
    split_merge_data()