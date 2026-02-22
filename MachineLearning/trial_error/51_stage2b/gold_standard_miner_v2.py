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
OUTPUT_FILE = 'gold_standard_data_v2.csv'

print("🚀 GOLD STANDARD MINER V2")
print("   Mencari fitur pembeda baru: Chroma Variation (Harmoni) & Spectral Flatness.")

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

# ================= 2. FEATURE EXTRACTION (NEW PHYSICS) =================
data_points = []

# Kumpulkan semua file
files = []
for d in SOURCE_DIRS:
    for m in ['sad', 'relaxed']:
        files.extend(glob.glob(os.path.join(d, m, '*.wav')) + glob.glob(os.path.join(d, m, '*.mp3')))

print(f"\n⏳ Menganalisis {len(files)} lagu...")

for path in tqdm(files):
    fid = os.path.basename(path).split('_')[0]
    if fid not in mood_map: continue
    
    current_label = mood_map[fid]
    lyric = lyrics_map.get(fid, "")
    
    # --- A. ANALISIS FISIKA AUDIO BARU ---
    try:
        y, sr = librosa.load(path, sr=16000)
        
        # 1. Chroma Variation (Seberapa sering nadanya berubah/kompleks?)
        # Lagu Relaxed biasanya looping (Variasi rendah)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_std = np.mean(np.std(chroma, axis=1))
        
        # 2. Spectral Flatness (Seberapa "bersih" atau "noisey" suaranya)
        # Lagu akustik (Sad) vs Synth/Lofi (Relaxed) punya flatness beda
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        
        # 3. Spectral Contrast (Tetap kita pakai karena tadi terbukti lumayan BEDA)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast)

    except:
        chroma_std, flatness_mean, contrast_mean = 0, 0, 0
        
    # --- B. ANALISIS TEKS ---
    score_sad = 0
    score_joy = 0
    
    if lyric and str(lyric).strip():
        text = re.sub(r"[^a-z0-9\s]", '', str(lyric).lower())[:512]
        try:
            res = nlp(text)[0]
            s = {item['label']: item['score'] for item in res}
            score_sad = s.get('sadness', 0) + s.get('fear', 0) 
            score_joy = s.get('joy', 0) + s.get('neutral', 0)
        except: pass
        
    data_points.append({
        'filename': os.path.basename(path),
        'original_label': current_label,
        'path': path,
        'chroma_std': chroma_std,   # Harmoni
        'flatness': flatness_mean,  # Kebisingan/Tone
        'contrast': contrast_mean,  # Tekstur
        'lyric_sad': score_sad,
        'lyric_joy': score_joy
    })

# Buat DataFrame
df_res = pd.DataFrame(data_points)

# ================= 3. SCORING & SELECTION =================
print("\n🧮 Menghitung Skor V2...")

scaler = MinMaxScaler()
cols = ['chroma_std', 'flatness', 'contrast', 'lyric_sad', 'lyric_joy']
df_norm = pd.DataFrame(scaler.fit_transform(df_res[cols]), columns=cols)

# --- RUMUS BARU ---
# Sad: Harmoni Kompleks (Chroma Tinggi) + Lirik Sedih + Contrast Tinggi
df_res['SAD_SCORE'] = df_norm['chroma_std'] + df_norm['contrast'] + (df_norm['lyric_sad'] * 2.0)

# Relaxed: Harmoni Stabil (Chroma Rendah) + Lirik Happy + Flatness Stabil
# (Note: Flatness di audio santai biasanya rendah/tonal, bukan noise)
df_res['RELAXED_SCORE'] = (1 - df_norm['chroma_std']) + (1 - df_norm['flatness']) + (df_norm['lyric_joy'] * 2.0)

# Ambil Top 20 (Kita persempit biar makin murni)
top_sad = df_res[df_res['original_label'] == 'sad'].nlargest(20, 'SAD_SCORE')
top_rel = df_res[df_res['original_label'] == 'relaxed'].nlargest(20, 'RELAXED_SCORE')

# ================= 4. EXPORT & CHECK =================
gold_df = pd.concat([top_sad, top_rel])
gold_df.to_csv(OUTPUT_FILE, index=False)

print("\n📊 VISUALISASI PERBEDAAN V2 (Rata-rata Kelompok Emas)")
print(f"{'Feature':<15} | {'SAD (Avg)':<10} | {'RELAXED (Avg)':<10} | {'BEDA':<10}")
print("-" * 55)

feat_list = ['chroma_std', 'flatness', 'contrast', 'lyric_sad', 'lyric_joy']
for f in feat_list:
    m_sad = top_sad[f].mean()
    m_rel = top_rel[f].mean()
    diff = abs(m_sad - m_rel)
    # Tanda centang jika beda > 10% dari nilai max (kira-kira)
    mark = "✅" if diff > 0.05 else "⚠️" 
    print(f"{f:<15} | {m_sad:.4f}     | {m_rel:.4f}     | {mark}")

print("-" * 55)
print("👉 Jika 'chroma_std' atau 'flatness' bertanda ✅, berarti ini fitur pengganti yang bagus!")
print(f"✅ Data tersimpan di '{OUTPUT_FILE}'")