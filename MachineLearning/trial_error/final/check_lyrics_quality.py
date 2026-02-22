import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONFIGURASI ---
FILE_PATH = 'data/lyrics/lyrics.xlsx'

print("🚀 MEMULAI PENGECEKAN KUALITAS DATA LIRIK...")
print("="*60)

# 1. LOAD DATA
if not os.path.exists(FILE_PATH):
    print(f"❌ File tidak ditemukan: {FILE_PATH}")
    exit()

try:
    df = pd.read_excel(FILE_PATH)
    print(f"✅ Berhasil memuat data. Total Baris: {len(df)}")
    print(f"📋 Kolom ditemukan: {list(df.columns)}")
except Exception as e:
    print(f"❌ Gagal membaca Excel: {e}")
    exit()

# Pastikan kolom lyrics ada
if 'lyrics' not in df.columns:
    print("❌ Kolom 'lyrics' tidak ditemukan! Cek nama kolom di Excel.")
    exit()

# Konversi ke string untuk keamanan
df['lyrics'] = df['lyrics'].astype(str)

print("-" * 60)

# --- 2. CEK KEKOSONGAN (NULL/EMPTY) ---
empty_lyrics = df[df['lyrics'].str.strip() == '']
nan_lyrics = df[df['lyrics'] == 'nan']
short_lyrics = df[df['lyrics'].str.len() < 50] # Lirik kurang dari 50 karakter

print("🔍 1. CEK KELENGKAPAN ISI:")
if len(empty_lyrics) > 0 or len(nan_lyrics) > 0:
    print(f"   ⚠️ PERINGATAN: Ada {len(empty_lyrics) + len(nan_lyrics)} baris lirik KOSONG/NaN.")
else:
    print("   ✅ Aman. Tidak ada lirik kosong.")

if len(short_lyrics) > 0:
    print(f"   ⚠️ PERINGATAN: Ada {len(short_lyrics)} lirik yang SANGAT PENDEK (< 50 karakter).")
    print(f"      Contoh ID: {short_lyrics['id'].head(3).tolist()}")
else:
    print("   ✅ Aman. Semua lirik memiliki panjang yang wajar.")


# --- 3. CEK "SAMPAH" METADATA (DIRTY TAGS) ---
print("\n🔍 2. CEK KEBERSIHAN (DIRTY TAGS):")

def count_patterns(pattern, name):
    count = df['lyrics'].str.contains(pattern, regex=True).sum()
    if count > 0:
        print(f"   ⚠️ Ditemukan {count} baris mengandung {name}")
    else:
        print(f"   ✅ Bersih dari {name}")

# Cek tag kurung siku [Chorus], [Verse]
count_patterns(r'\[.*?\]', 'Tag Metadata [Chorus/Verse]')
# Cek tag kurung biasa (x2), (Live)
count_patterns(r'\(.*?\)', 'Tag Kurung Biasa (x...)')
# Cek kata-kata sampah website
count_patterns(r'(?i)embed|lyrics|contributor|ticket', 'Kata Sampah Web (Embed/Lyrics)')
# Cek karakter non-ASCII (Bahasa asing aneh/Emoji)
count_patterns(r'[^\x00-\x7F]+', 'Karakter Non-ASCII/Emoji')


# --- 4. CEK DISTRIBUSI MOOD ---
print("\n🔍 3. CEK KESEIMBANGAN KELAS (MOOD):")
if 'mood' in df.columns:
    mood_counts = df['mood'].value_counts()
    print(mood_counts)
    
    # Visualisasi Bar Chart
    plt.figure(figsize=(8, 4))
    sns.barplot(x=mood_counts.index, y=mood_counts.values, palette='viridis')
    plt.title("Distribusi Jumlah Lagu per Mood")
    plt.ylabel("Jumlah Lagu")
    plt.xlabel("Mood")
    plt.show()
else:
    print("   ⚠️ Kolom 'mood' tidak ditemukan.")


# --- 5. STATISTIK PANJANG KATA ---
print("\n🔍 4. STATISTIK PANJANG KATA:")
df['word_count'] = df['lyrics'].apply(lambda x: len(str(x).split()))

avg_words = df['word_count'].mean()
min_words = df['word_count'].min()
max_words = df['word_count'].max()

print(f"   📝 Rata-rata kata per lagu : {avg_words:.1f} kata")
print(f"   📝 Lagu terpendek          : {min_words} kata")
print(f"   📝 Lagu terpanjang         : {max_words} kata")

# Visualisasi Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['word_count'], bins=30, kde=True, color='skyblue')
plt.title("Distribusi Panjang Lirik (Jumlah Kata)")
plt.xlabel("Jumlah Kata")
plt.axvline(avg_words, color='red', linestyle='--', label=f'Rata-rata: {avg_words:.0f}')
plt.legend()
plt.show()


# --- 6. SAMPLE CHECK ---
print("\n🔍 5. CONTOH SAMPEL ACAK (Cek Manual Mata):")
print("-" * 60)
sample = df.sample(3)
for idx, row in sample.iterrows():
    print(f"🎵 ID: {row.get('id', 'N/A')} | Mood: {row.get('mood', 'N/A')}")
    print(f"📝 Cuplikan Lirik (100 char pertama):\n   {str(row['lyrics'])[:100]}...")
    print("-" * 30)

print("\n✅ PENGECEKAN SELESAI.")