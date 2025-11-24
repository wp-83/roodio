import pandas as pd
import lyricsgenius
import time
import os
from tqdm import tqdm
import src.config as cfg

# ================= KONFIGURASI =================
# GANTI DENGAN TOKEN ANDA DARI GENIUS (Wajib Diisi)
GENIUS_TOKEN = "xP48Tl1sk5qHyX_vi9No0Gia4e5JH-LYMd8YS9UX5dJ1ztftJBun_g-7ELssYHIl"

# Folder Metadata & Output
METADATA_FOLDER = os.path.join(cfg.BASE_DIR, "data", "metadata", "lyrics")
OUTPUT_FILE = os.path.join(cfg.BASE_DIR, "data", "processed", "lyrics", "lyrics_dataset_raw.csv")
# ===============================================

def load_and_merge_metadata():
    """
    Mencari semua file CSV di folder metadata, 
    mendeteksi format kolom yang berbeda-beda, 
    dan menggabungkannya menjadi satu standar.
    """
    print("📂 Membaca dan Menstandarkan Metadata...")
    
    # Cek apakah folder ada
    if not os.path.exists(METADATA_FOLDER):
        raise FileNotFoundError(f"Folder tidak ditemukan: {METADATA_FOLDER}")

    # Gunakan os.listdir (Aman untuk path dengan karakter spesial seperti [])
    all_files = os.listdir(METADATA_FOLDER)
    csv_files = [os.path.join(METADATA_FOLDER, f) for f in all_files if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file CSV di {METADATA_FOLDER}")
    
    df_list = []
    
    # Kolom standar yang kita inginkan untuk output
    target_columns = ['song_id', 'Artist', 'Title']

    for filename in csv_files:
        try:
            # Baca CSV dengan toleransi error (skip bad lines)
            df = pd.read_csv(filename, on_bad_lines='skip', engine='python')
            
            # Bersihkan nama kolom (hapus spasi depan/belakang)
            df.columns = df.columns.str.strip()
            
            # --- LOGIKA DETEKSI & RENAME KOLOM ---
            
            # KASUS 1: Metadata 2014 (Id, Artist, Track)
            if 'Id' in df.columns and 'Track' in df.columns:
                print(f"   - Format 2014 terdeteksi: {os.path.basename(filename)}")
                df = df.rename(columns={
                    'Id': 'song_id', 
                    'Track': 'Title'
                })
            
            # KASUS 2: Metadata 2015 (id, artist, title) - Huruf kecil
            elif 'id' in df.columns and 'title' in df.columns:
                print(f"   - Format 2015 terdeteksi: {os.path.basename(filename)}")
                df = df.rename(columns={
                    'id': 'song_id',
                    'artist': 'Artist',
                    'title': 'Title'
                })

            # KASUS 3: Metadata 2013 (song_id, Artist, Song title)
            elif 'Song title' in df.columns:
                print(f"   - Format 2013 terdeteksi: {os.path.basename(filename)}")
                df = df.rename(columns={
                    'Song title': 'Title'
                })
                
            # Cek apakah kolom standar sudah lengkap setelah rename?
            if all(col in df.columns for col in target_columns):
                # Ambil hanya kolom yang dibutuhkan agar rapi
                df_clean = df[target_columns]
                df_list.append(df_clean)
            else:
                # Jika masih tidak cocok, skip file ini (mungkin file lain/kamus)
                # print(f"   ⚠️ Skip {os.path.basename(filename)}: Format tidak cocok.")
                pass
                
        except Exception as e:
            print(f"   ! Gagal baca {os.path.basename(filename)}: {e}")

    if not df_list:
        raise ValueError("Tidak ada data metadata yang berhasil diproses!")

    # Gabung Semua DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Hapus duplikat berdasarkan ID lagu
    merged_df = merged_df.drop_duplicates(subset=['song_id'])
    
    # Rename kolom agar sesuai dengan script scraping (Artist & Song title)
    # Kita kembalikan 'Title' menjadi 'Song title' agar konsisten
    merged_df = merged_df.rename(columns={'Title': 'Song title'})
    
    print(f"✅ Total lagu unik siap diproses: {len(merged_df)}")
    return merged_df

def scrape_lyrics():
    # 1. Setup Genius
    if "MASUKKAN" in GENIUS_TOKEN:
        print("❌ ERROR: Harap isi GENIUS_TOKEN di dalam script terlebih dahulu!")
        return

    print(f"🔌 Menghubungkan ke Genius API...")
    genius = lyricsgenius.Genius(GENIUS_TOKEN)
    genius.verbose = False 
    genius.remove_section_headers = True 
    genius.skip_non_songs = True
    
    # 2. Load & Merge Metadata
    try:
        df_metadata = load_and_merge_metadata()
    except Exception as e:
        print(f"❌ Error Metadata: {e}")
        return

    # 3. Cek Resume (Apakah sudah ada file sebelumnya?)
    # Buat folder output jika belum ada
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            # Cek kolom 'song_id' ada atau tidak
            if 'song_id' in existing_df.columns:
                processed_ids = existing_df['song_id'].unique()
                # Filter lagu yang belum diproses
                df_metadata = df_metadata[~df_metadata['song_id'].isin(processed_ids)]
                print(f"⚠️ Resume: Melanjutkan sisa {len(df_metadata)} lagu...")
            else:
                print("⚠️ File output ada tapi format salah. Mulai dari awal.")
                pd.DataFrame(columns=['song_id', 'artist', 'title', 'lyrics']).to_csv(OUTPUT_FILE, index=False)
        except:
            print("⚠️ File output rusak. Mulai dari awal.")
            pd.DataFrame(columns=['song_id', 'artist', 'title', 'lyrics']).to_csv(OUTPUT_FILE, index=False)
    else:
        # Buat file baru dengan header
        pd.DataFrame(columns=['song_id', 'artist', 'title', 'lyrics']).to_csv(OUTPUT_FILE, index=False)

    # 4. Mulai Scraping
    print("\n🚀 Memulai Scraping (Tekan Ctrl+C untuk berhenti)...")
    
    songs_list = df_metadata.to_dict('records')

    for i, song in enumerate(tqdm(songs_list)):
        song_id = song['song_id']
        artist = str(song['Artist']).strip()
        title = str(song['Song title']).strip()
        
        if not artist or not title or artist.lower() == 'nan':
            continue

        try:
            # Cari Lirik
            song_data = genius.search_song(title, artist)
            
            lyric_text = ""
            if song_data:
                lyric_text = song_data.lyrics
            
            # Simpan baris baru
            new_row = {
                'song_id': song_id,
                'artist': artist,
                'title': title,
                'lyrics': lyric_text
            }
            
            # Append ke CSV (Save per lagu)
            pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
            # Jeda Waktu (Rate Limiting)
            time.sleep(3 if lyric_text else 1)

        except Exception as e:
            print(f"\n❌ Error pada {artist} - {title}: {e}")
            time.sleep(5) 

    print("\n✅ Proses Selesai! Cek file di:", OUTPUT_FILE)

if __name__ == "__main__":
    scrape_lyrics()