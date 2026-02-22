import os
import shutil
import pandas as pd
import glob
from tqdm import tqdm

# ================= CONFIGURATION =================
# Folder sumber (Raw Data yang kotor)
SOURCE_DIRS = ['data/raw', 'data/raw2']

# File Filter (Daftar Lagu Emas)
GOLD_CSV = 'gold_standard_trifecta.csv'

# Folder Tujuan (Dataset Bersih)
DEST_DIR = 'data/FINAL_DATASET_CLEAN'

# Kategori
MOODS = ['angry', 'happy', 'sad', 'relaxed']

print("🚀 DATASET CLEANER & MERGER")
print("   Tujuan: Membuat folder dataset baru yang hanya berisi data berkualitas.")
print(f"   Target Folder: {DEST_DIR}")
print("="*60)

# ================= 1. PREPARATION =================

# Load Daftar Gold Standard
if not os.path.exists(GOLD_CSV):
    print(f"❌ Error: {GOLD_CSV} tidak ditemukan.")
    exit()

df_gold = pd.read_csv(GOLD_CSV)
# Kita buat Set berisi filename agar pencarian cepat
gold_filenames = set(df_gold['filename'].tolist())

print(f"✅ Gold Standard Loaded: {len(gold_filenames)} files (Sad/Relaxed).")

# Buat Folder Tujuan
if os.path.exists(DEST_DIR):
    print(f"⚠️  Folder '{DEST_DIR}' sudah ada. Menghapus untuk reset...")
    shutil.rmtree(DEST_DIR)

os.makedirs(DEST_DIR)
for m in MOODS:
    os.makedirs(os.path.join(DEST_DIR, m))

# ================= 2. COPY PROCESS =================

stats = {m: 0 for m in MOODS}
skipped = 0

print("\n📦 Copying Files...")

for src_dir in SOURCE_DIRS:
    for mood in MOODS:
        # Cari semua file audio di folder sumber
        src_files = glob.glob(os.path.join(src_dir, mood, '*.wav')) + \
                    glob.glob(os.path.join(src_dir, mood, '*.mp3'))
        
        for file_path in tqdm(src_files, desc=f"Processing {src_dir}/{mood}", leave=False):
            filename = os.path.basename(file_path)
            
            # --- LOGIKA PENYARINGAN ---
            should_copy = False
            
            if mood in ['angry', 'happy']:
                # Angry & Happy: COPY SEMUA (Sesuai request)
                should_copy = True
            
            elif mood in ['sad', 'relaxed']:
                # Sad & Relaxed: COPY HANYA JIKA ADA DI GOLD CSV
                if filename in gold_filenames:
                    should_copy = True
                else:
                    should_copy = False # Buang (Sampah)
            
            # --- EKSEKUSI COPY ---
            if should_copy:
                dest_path = os.path.join(DEST_DIR, mood, filename)
                try:
                    shutil.copy2(file_path, dest_path) # Copy file + metadata
                    stats[mood] += 1
                except Exception as e:
                    print(f"❌ Gagal copy {filename}: {e}")
            else:
                skipped += 1

# ================= 3. FINAL REPORT =================
print("\n" + "="*60)
print("📊 FINAL DATASET REPORT")
print("="*60)
print(f"📁 Lokasi Dataset Baru : {DEST_DIR}")
print("-" * 60)
print("Jumlah File per Kategori:")
for m in MOODS:
    print(f"   - {m.upper():<10} : {stats[m]} files")

print("-" * 60)
print(f"✅ Total File Bersih : {sum(stats.values())}")
print(f"🗑️  Total Sampah Dibuang: {skipped} (Sad/Relaxed yang tidak Gold)")
print("="*60)
print("👉 Sekarang gunakan folder 'data/FINAL_DATASET_CLEAN' untuk training final!")