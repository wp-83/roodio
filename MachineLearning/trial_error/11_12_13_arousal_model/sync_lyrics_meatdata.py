import pandas as pd
from pathlib import Path

# =====================
# PATH
# =====================
lyrics_csv = Path(r"data/lyrics/fixed_one_row.csv")
meta_csv   = Path(r"data/lyrics/metadata_lyrics.csv")
output_csv = Path(r"data/lyrics/final_dataset.csv")

# =====================
# LOAD
# =====================
df_lyrics = pd.read_csv(lyrics_csv, sep=';')
df_meta   = pd.read_csv(meta_csv, sep=';')

# =====================
# NORMALISASI
# =====================
df_lyrics.columns = df_lyrics.columns.str.strip().str.lower()
df_meta.columns   = df_meta.columns.str.strip().str.lower()

# =====================
# GABUNG PER MOOD + URUTAN
# =====================
final_rows = []

for mood in df_lyrics['mood'].unique():
    lyrics_mood = df_lyrics[df_lyrics['mood'] == mood].reset_index(drop=True)
    meta_mood   = df_meta[df_meta['mood'] == mood].reset_index(drop=True)

    assert len(lyrics_mood) == len(meta_mood), \
        f"Jumlah lagu mood '{mood}' tidak sama!"

    for i in range(len(lyrics_mood)):
        final_rows.append({
            "lyrics": lyrics_mood.loc[i, 'lyrics'],
            "mood": mood,
            "title": meta_mood.loc[i, 'title'],
            "artist": meta_mood.loc[i, 'artist']
        })

# =====================
# SAVE
# =====================
df_final = pd.DataFrame(final_rows)
df_final.to_csv(output_csv, sep=';', index=False)

print("✅ DATASET FINAL BERHASIL DIBUAT")
print(df_final.head())
