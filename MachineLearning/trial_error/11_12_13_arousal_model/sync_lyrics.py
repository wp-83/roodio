from pathlib import Path
import pandas as pd

input_csv = Path(r"data\lyrics\lyrics copy.csv")
output_csv = Path(r"data\lyrics\fixed_one_row.csv")

songs = []
current_lyrics = []

with open(input_csv, "r", encoding="utf-8") as f:
    lines = f.readlines()

# skip header
for line in lines[1:]:
    line = line.strip()

    # BARIS TERAKHIR LAGU (ADA ;mood)
    if ";" in line:
        lyric_part, mood = line.rsplit(";", 1)
        current_lyrics.append(lyric_part)

        full_lyrics = " ".join(current_lyrics)
        full_lyrics = " ".join(full_lyrics.split())

        songs.append({
            "lyrics": full_lyrics,
            "mood": mood
        })

        current_lyrics = []  # reset untuk lagu berikutnya
    else:
        current_lyrics.append(line)

# SIMPAN KE CSV BENAR
df_final = pd.DataFrame(songs)
df_final.to_csv(output_csv, sep=";", index=False)

print("✅ DONE: CSV FIXED DENGAN BENAR")
print(df_final.head())
