import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
OUTPUT_FILE = 'gold_standard_trifecta.csv'

print("🚀 GOLD MINER: TRIFECTA EDITION")
print("   Mencari data terbaik menggunakan 3 Fitur Kunci: Contrast, Sadness, Joy.")

# ================= 1. LOAD METADATA =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    mood_map = dict(zip(df['id'], df['mood']))
except Exception as e:
    print(f"❌ Error Excel: {e}")
    exit()

nlp = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

# ================= 2. FEATURE EXTRACTION (ONLY 3) =================
data_points = []

# Kumpulkan semua file
files = []
for d in SOURCE_DIRS:
    for m in ['sad', 'relaxed']:
        files.extend(glob.glob(os.path.join(d, m, '*.wav')) + glob.glob(os.path.join(d, m, '*.mp3')))

print(f"\n⏳ Mining {len(files)} lagu...")

for path in tqdm(files):
    fid = os.path.basename(path).split('_')[0]
    if fid not in mood_map: continue
    
    current_label = mood_map[fid]
    lyric = lyrics_map.get(fid, "")
    
    # --- A. AUDIO: SPECTRAL CONTRAST (Tekstur) ---
    try:
        y, sr = librosa.load(path, sr=16000)
        # Contrast tinggi = Kasar/Tajam (Sad)
        # Contrast rendah = Halus/Smooth (Relaxed)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast)
    except:
        contrast_mean = 0
        
    # --- B. TEXT: SADNESS & JOY ---
    score_sad = 0
    score_joy = 0
    
    if lyric and str(lyric).strip():
        text = re.sub(r"[^a-z0-9\s]", '', str(lyric).lower())[:512]
        try:
            res = nlp(text)[0]
            s = {item['label']: item['score'] for item in res}
            # Gabung Fear ke Sadness
            score_sad = s.get('sadness', 0) + s.get('fear', 0) 
            # Gabung Neutral ke Joy
            score_joy = s.get('joy', 0) + s.get('neutral', 0)
        except: pass
        
    data_points.append({
        'filename': os.path.basename(path),
        'original_label': current_label,
        'path': path,
        'contrast': contrast_mean,  # Fitur Audio Tunggal
        'lyric_sad': score_sad,     # Fitur Teks 1
        'lyric_joy': score_joy      # Fitur Teks 2
    })

# Buat DataFrame
df_res = pd.DataFrame(data_points)

# ================= 3. SCORING ALGORITHM (TRIFECTA) =================
print("\n🧮 Menghitung Skor Emas...")

# Normalisasi data (0-1) agar Audio (20-an) setara dengan Teks (0-1)
scaler = MinMaxScaler()
cols = ['contrast', 'lyric_sad', 'lyric_joy']
df_norm = pd.DataFrame(scaler.fit_transform(df_res[cols]), columns=cols)

# --- RUMUS SKOR BARU ---

# Skor SAD Ideal:
# - Contrast Tinggi (Suara Tajam/Kasar)
# - Lirik Sedih Tinggi
df_res['SAD_SCORE'] = df_norm['contrast'] + (df_norm['lyric_sad'] * 2.0) 

# Skor RELAXED Ideal:
# - Contrast Rendah (Suara Halus/Smooth) -> (1 - contrast)
# - Lirik Happy/Netral Tinggi
df_res['RELAXED_SCORE'] = (1 - df_norm['contrast']) + (df_norm['lyric_joy'] * 2.0)

# ================= 4. SELECT TOP DATA =================

# Ambil Top 20 Sad Terbaik
top_sad = df_res[df_res['original_label'] == 'sad'].nlargest(35, 'SAD_SCORE')

# Ambil Top 20 Relaxed Terbaik
top_rel = df_res[df_res['original_label'] == 'relaxed'].nlargest(35, 'RELAXED_SCORE')

# ================= 5. REPORT & EXPORT =================

print("\n" + "="*80)
print("💎 TRIFECTA GOLD: SAD (Top 5)")
print("   Ciri: Lirik Galau + Audio Tajam/Kontras Tinggi")
print("="*80)
print(top_sad[['filename', 'SAD_SCORE', 'contrast', 'lyric_sad']].head(5).to_string(index=False))

print("\n" + "="*80)
print("💎 TRIFECTA GOLD: RELAXED (Top 5)")
print("   Ciri: Lirik Happy/Tenang + Audio Halus/Kontras Rendah")
print("="*80)
print(top_rel[['filename', 'RELAXED_SCORE', 'contrast', 'lyric_joy']].head(5).to_string(index=False))

# Gabungkan & Simpan
gold_df = pd.concat([top_sad, top_rel])
gold_df.to_csv(OUTPUT_FILE, index=False)

print("\n📊 VISUALISASI FINAL (Rata-rata)")
print(f"{'Feature':<15} | {'SAD (Avg)':<10} | {'RELAXED (Avg)':<10} | {'BEDA':<10}")
print("-" * 55)

feat_list = ['contrast', 'lyric_sad', 'lyric_joy']
for f in feat_list:
    m_sad = top_sad[f].mean()
    m_rel = top_rel[f].mean()
    diff = abs(m_sad - m_rel)
    mark = "✅" if diff > 0.05 else "⚠️" 
    print(f"{f:<15} | {m_sad:.4f}     | {m_rel:.4f}     | {mark}")

print("-" * 55)
print(f"✅ Data Tersimpan: '{OUTPUT_FILE}'")
print("   Gunakan file ini untuk training model Final Trifecta!")