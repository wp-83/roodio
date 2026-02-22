import pandas as pd
import matplotlib.pyplot as plt
import os

# Lokasi file CSV hasil ekstraksi fitur (sesuaikan path-nya)
DATA_PATH = 'data/processed/features.csv' 

def cek_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan.")
        print("   Pastikan Anda sudah menjalankan ekstraksi fitur (main.py / extract.py) dulu.")
        return

    print("📊 Membaca data...")
    df = pd.read_csv(DATA_PATH)
    
    # Hitung jumlah per label
    counts = df['label'].value_counts()
    
    print("\n--- JUMLAH LAGU PER KATEGORI ---")
    print(counts)
    print("--------------------------------")
    
    # Cek Imbalance
    max_val = counts.max()
    min_val = counts.min()
    ratio = max_val / min_val
    
    if ratio > 1.5:
        print(f"⚠️ PERINGATAN: Data tidak seimbang! (Rasio {ratio:.1f}x)")
        print("   Kategori terbanyak mendominasi kategori tersedikit.")
        print("   Solusi: Tambah lagu untuk kategori yang sedikit.")
    else:
        print("✅ Status: Data cukup seimbang.")

    # Plot Grafik
    plt.figure(figsize=(8, 5))
    counts.plot(kind='bar', color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
    plt.title('Distribusi Jumlah Lagu per Mood')
    plt.xlabel('Mood')
    plt.ylabel('Jumlah Sampel')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cek_data()