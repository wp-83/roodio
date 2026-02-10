import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from tqdm import tqdm
import time
import os

# ================= CONFIG =================
# Isi dengan Kunci Spotify Developer Kamu
CLIENT_ID = '78d79ca07730480d9cffaeb2db392f29'
CLIENT_SECRET = '3a008c70357c49ecb66486a53bdc76bd'

# Input: Excel Lirik (Wajib ada kolom 'id' dan 'lyrics')
LYRICS_FILE = 'data/lyrics/lyrics.xlsx' 

# Output: CSV Lengkap
OUTPUT_CSV = 'dataset_regresi_master.csv'

# ================= SETUP =================
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

def get_spotify_data(artist, title):
    """Mencari Valence berdasarkan Artis dan Judul"""
    query = f"artist:{artist} track:{title}"
    try:
        results = sp.search(q=query, type='track', limit=1)
        items = results['tracks']['items']
        if items:
            track = items[0]
            tid = track['id']
            meta = sp.audio_features([tid])[0]
            if meta:
                return meta['valence'], track['name'], track['artists'][0]['name']
    except:
        pass
    return None, None, None

# ================= MAIN LOOP =================
print("🚀 MEMBUAT MASTER DATASET (Artist + Lirik + Valence)...")

if not os.path.exists(LYRICS_FILE):
    print(f"❌ File {LYRICS_FILE} tidak ditemukan!")
    exit()

df = pd.read_excel(LYRICS_FILE)
print(f"📄 Membaca {len(df)} data lirik lokal.")

final_data = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # 1. Parsing Nama File -> Artist & Title
    # Asumsi format ID: "Artist - Title.mp3" atau "Artist - Title"
    filename = str(row['id']).replace('.mp3', '').replace('.wav', '')
    lyrics_text = str(row['lyrics']) if pd.notna(row['lyrics']) else ""
    
    # Logika Pemisahan (Split)
    if ' - ' in filename:
        parts = filename.split(' - ', 1)
        artist_local = parts[0].strip()
        title_local = parts[1].strip()
    else:
        # Jika format tidak baku, anggap nama file sebagai judul, artis kosong
        artist_local = "" 
        title_local = filename.strip()

    # 2. Cari di Spotify
    valence, sp_title, sp_artist = get_spotify_data(artist_local, title_local)
    
    # 3. Simpan Data (Hanya jika ketemu di Spotify)
    if valence is not None:
        final_data.append({
            'filename': str(row['id']),       # Link ke file audio
            'artist': sp_artist,              # Nama Artis (dari Spotify biar rapi)
            'title': sp_title,                # Judul Lagu
            'lyrics': lyrics_text,            # Lirik (WAJIB ADA)
            'valence_target': valence         # Target Regresi (0.0 - 1.0)
        })
    else:
        # Opsional: Jika tidak ketemu, bisa skip atau simpan dengan valence kosong
        # Di sini kita skip agar dataset bersih untuk training
        pass
        
    # Jeda biar API tidak error
    time.sleep(0.1)

# Simpan ke CSV
df_final = pd.DataFrame(final_data)
df_final.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*50)
print(f"✅ DATASET SELESAI!")
print(f"📊 Total Lagu: {len(df_final)}")
print(f"💾 Disimpan ke: {OUTPUT_CSV}")
print("   (Kolom: filename, artist, title, lyrics, valence_target)")