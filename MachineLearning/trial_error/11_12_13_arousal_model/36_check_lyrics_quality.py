import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
LYRICS_PATH = 'data/lyrics/lyrics1.csv'
MIN_WORD_COUNT = 10 # Ambang batas lirik yang dianggap "Valid"

print(f"🚀 MEMULAI EXP 36: LYRICS HEALTH CHECK...\n")

# --- 1. LOAD DATA ---
if not os.path.exists(LYRICS_PATH):
    print("❌ File CSV tidak ditemukan!")
    exit()

try:
    # Coba baca smart engine
    df = pd.read_csv(LYRICS_PATH, sep=None, engine='python')
    print(f"✅ Berhasil membaca CSV. Separator terdeteksi otomatis.")
except:
    try:
        df = pd.read_csv(LYRICS_PATH, sep=';')
        print(f"✅ Berhasil membaca CSV dengan separator ';'")
    except:
        df = pd.read_csv(LYRICS_PATH, sep=',')
        print(f"✅ Berhasil membaca CSV dengan separator ','")

# Normalisasi Header
df.columns = df.columns.str.strip().str.lower()
print(f"📋 Kolom ditemukan: {df.columns.tolist()}")

if 'lyrics' not in df.columns or 'mood' not in df.columns:
    print("❌ ERROR: Kolom 'lyrics' atau 'mood' tidak ditemukan!")
    exit()

# --- 2. CEK TOTAL DATA & FORMAT ---
total_rows = len(df)
print(f"\n📊 TOTAL BARIS DATA: {total_rows}")

# Cek Label Mood
print("\n🔍 1. CEK KONSISTENSI LABEL MOOD:")
print(df['mood'].value_counts())
# Normalisasi label
df['mood_clean'] = df['mood'].astype(str).str.lower().str.strip()
print("   (Normalisasi menjadi lowercase & trim whitespace...)")

# --- 3. CEK KEKOSONGAN (NULL) ---
print("\n🔍 2. CEK DATA KOSONG (NULL):")
null_lyrics = df['lyrics'].isnull().sum()
if null_lyrics > 0:
    print(f"   ⚠️ PERINGATAN: Ada {null_lyrics} baris dengan lirik KOSONG (NaN)!")
    print(df[df['lyrics'].isnull()])
else:
    print("   ✅ Aman. Tidak ada lirik NaN.")

# --- 4. CEK ISI SAMPAH (GARBAGE CONTENT) ---
print("\n🔍 3. CEK KUALITAS ISI LIRIK:")

# Hitung jumlah kata
df['word_count'] = df['lyrics'].astype(str).apply(lambda x: len(x.split()))

# Filter lirik kependekan
short_lyrics = df[df['word_count'] < MIN_WORD_COUNT]

if len(short_lyrics) > 0:
    print(f"   ⚠️ PERINGATAN: Ada {len(short_lyrics)} lagu dengan lirik < {MIN_WORD_COUNT} kata.")
    print("   (Ini biasanya Instrumental atau lirik error. Sebaiknya DIHAPUS).")
    print("-" * 60)
    print(f"   {'Mood':<10} | {'WordCount':<10} | {'Snippet (50 chars)'}")
    print("-" * 60)
    for idx, row in short_lyrics.iterrows():
        snippet = str(row['lyrics'])[:50].replace('\n', ' ')
        print(f"   {row['mood_clean']:<10} | {row['word_count']:<10} | {snippet}...")
else:
    print("   ✅ Aman. Semua lirik cukup panjang.")

# Cek Keyword Berbahaya
keywords = ['instrumental', 'tidak ada lirik', 'no lyrics', 'lyrics not found']
garbage_indices = []
for idx, row in df.iterrows():
    txt = str(row['lyrics']).lower()
    for k in keywords:
        if k in txt and len(txt) < 50: # Kalau ada kata ini dan pendek
            garbage_indices.append(idx)

if len(garbage_indices) > 0:
    print(f"\n   ⚠️ PERINGATAN: Ditemukan {len(garbage_indices)} baris terindikasi SAMPAH (Instrumental/Error):")
    print(df.loc[garbage_indices, ['mood', 'lyrics']])

# --- 5. VISUALISASI DISTRIBUSI KATA ---
print("\n📊 Statistik Panjang Lirik:")
print(df.groupby('mood_clean')['word_count'].describe())

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='word_count', hue='mood_clean', kde=True, bins=20)
plt.title('Distribusi Panjang Lirik per Mood')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.axvline(x=MIN_WORD_COUNT, color='red', linestyle='--', label='Batas Minimal')
plt.legend()
plt.tight_layout()
plt.savefig('lyrics_quality_check.png')
print("\n📈 Grafik distribusi disimpan ke 'lyrics_quality_check.png'")

# --- 6. REKOMENDASI PERBAIKAN ---
print("\n" + "="*60)
print("🛠️ REKOMENDASI PERBAIKAN:")
print("="*60)

rekomendasi = []
if null_lyrics > 0: rekomendasi.append("- Hapus baris yang liriknya kosong.")
if len(short_lyrics) > 0: rekomendasi.append(f"- Hapus {len(short_lyrics)} lagu yang liriknya terlalu pendek (kemungkinan instrumental).")
if len(df['mood_clean'].unique()) > 4: rekomendasi.append("- Perbaiki penulisan label mood di Excel (misal: 'Sad ' menjadi 'sad').")

if len(rekomendasi) == 0:
    print("✅ DATA SUDAH BERSIH! Kualitas input bagus.")
else:
    for r in rekomendasi:
        print(r)
    print("\n👉 Bersihkan file Excel/CSV Anda manual atau gunakan dropna() sebelum training.")

print("-" * 60)