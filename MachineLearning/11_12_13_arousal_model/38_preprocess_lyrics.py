import pandas as pd
import re
import os

# --- KONFIGURASI ---
INPUT_FILE = 'data/lyrics/lyrics1.csv'   # File hasil perbaikan manual kamu
OUTPUT_FILE = 'data/lyrics/lyrics_cleaned.csv' # File siap training

print(f"🚀 MEMULAI EXP 38: LYRICS CLEANING & PREPROCESSING...\n")

# --- 1. LOAD DATA ---
if not os.path.exists(INPUT_FILE):
    print(f"❌ File {INPUT_FILE} tidak ditemukan!")
    exit()

df = pd.read_csv(INPUT_FILE, sep=None, engine='python')
print(f"📊 Total Data Awal: {len(df)} baris")

# --- 2. FUNGSI CLEANING ---
def clean_lyrics_text(text):
    if pd.isna(text) or text == '':
        return ""
    
    # 1. Lowercase
    text = str(text).lower()
    
    # 2. Hapus Metadata dalam kurung siku/biasa (Contoh: [Chorus], (x2), [Verse 1])
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    
    # 3. Hapus kata-kata sampah umum dari web lirik
    garbage_words = ['lyrics', 'embed', 'contributors', 'click to see', 'translation']
    for word in garbage_words:
        text = text.replace(word, '')
        
    # 4. Ganti karakter baris baru (\n) dengan spasi agar jadi satu paragraf utuh
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 5. Hapus karakter non-ASCII (Emoji, simbol aneh), TAPI pertahankan tanda baca penting (. , ! ?)
    # Regex ini berarti: Hapus apa pun KECUALI huruf (a-z), angka (0-9), dan tanda baca basic
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text)
    
    # 6. Hapus spasi berlebih (double spaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- 3. EKSEKUSI CLEANING ---
print("🧹 Membersihkan lirik...")
df['lyrics_clean'] = df['lyrics'].apply(clean_lyrics_text)

# --- 4. VALIDASI HASIL ---
# Cek apakah ada lirik yang jadi kosong setelah dibersihkan
empty_after_clean = df[df['lyrics_clean'].str.len() < 5]
if len(empty_after_clean) > 0:
    print(f"⚠️ PERINGATAN: Ada {len(empty_after_clean)} lirik menjadi kosong/terlalu pendek setelah cleaning!")
    print(empty_after_clean[['filename', 'lyrics', 'lyrics_clean']])
else:
    print("✅ Semua lirik aman (tidak ada yang kosong).")

# --- 5. PERBANDINGAN SEBELUM VS SESUDAH ---
print("\n🔍 CONTOH HASIL CLEANING:")
sample = df.sample(1).iloc[0]
print("-" * 50)
print(f"🎵 Judul/File: {sample.get('filename', 'Unknown')}")
print("-" * 50)
print("❌ ORIGINAL:")
print(str(sample['lyrics'])[:300] + "...")
print("-" * 50)
print("✅ CLEANED:")
print(str(sample['lyrics_clean'])[:300] + "...")
print("-" * 50)

# --- 6. SIMPAN FILE BARU ---
# Kita hanya simpan kolom yang bersih untuk training nanti
final_df = df.copy()
final_df['lyrics'] = final_df['lyrics_clean'] # Timpa kolom lama dengan yang bersih
# Jika ada kolom lain yang tidak perlu, bisa di-drop, tapi biarkan saja dulu.

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Data bersih disimpan ke: {OUTPUT_FILE}")
print("👉 Gunakan file ini untuk Experiment 37 (Training)!")