import os
import logging

# Mute Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# --- KONFIGURASI ---
LYRICS_PATH = 'data/lyrics/lyrics.csv'
TARGET_MOODS = ['angry', 'happy'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

print(f"🚀 MEMULAI EXP 24: DIAGNOSA EMOSI DOMINAN...")

# --- 1. LOAD DATA ---
if not os.path.exists(LYRICS_PATH):
    print("❌ File tidak ditemukan.")
    exit()

try:
    df = pd.read_csv(LYRICS_PATH, sep=';')
    if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',')
    df.columns = df.columns.str.strip().str.lower()
except: exit()

# Filter Data
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()

# --- 2. LOAD MODEL ---
classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 3. DIAGNOSTIC LOOP ---
print(f"🧠 Menganalisis {len(df)} lagu untuk melihat EMOSI JUARA 1...")

results_log = []

for index, row in tqdm(df.iterrows(), total=len(df)):
    lyrics = str(row['lyrics'])
    true_mood = row['mood']
    title = row['title'] if 'title' in df.columns else f"Song {index}"
    
    if len(lyrics) < 2: continue

    try:
        # Inference
        output = classifier(lyrics)[0] # List of dicts [{'label': 'joy', 'score': 0.9}, ...]
        
        # Cari Emosi dengan Skor TERTINGGI (Juara 1)
        # Sort berdasarkan score descending
        sorted_emotions = sorted(output, key=lambda k: k['score'], reverse=True)
        
        top_emotion = sorted_emotions[0]['label'] # Misal: 'neutral'
        top_score = sorted_emotions[0]['score']   # Misal: 0.85
        
        # Cari skor Joy dan Anger juga buat perbandingan
        joy_score = next(item['score'] for item in output if item['label'] == 'joy')
        anger_score = next(item['score'] for item in output if item['label'] == 'anger')

        results_log.append({
            'Title': title,
            'True_Mood': true_mood,
            'AI_Thinks': top_emotion,     # <--- INI YANG KITA CARI
            'AI_Conf': top_score,
            'Joy_Score': joy_score,
            'Anger_Score': anger_score
        })
        
    except Exception as e:
        print(f"Error {title}: {e}")

# --- 4. LAPORAN DIAGNOSA ---
df_res = pd.DataFrame(results_log)

# Pisahkan per Mood Asli
angry_songs = df_res[df_res['True_Mood'] == 'angry']
happy_songs = df_res[df_res['True_Mood'] == 'happy']

print("\n" + "="*60)
print("🧐 HASIL DIAGNOSA: Lagu ANGRY Sebenarnya Terdeteksi Apa?")
print("="*60)
print(angry_songs['AI_Thinks'].value_counts())

print("\n" + "="*60)
print("🧐 HASIL DIAGNOSA: Lagu HAPPY Sebenarnya Terdeteksi Apa?")
print("="*60)
print(happy_songs['AI_Thinks'].value_counts())

print("\n" + "="*60)
print("❌ DETAIL LAGU YANG TIDAK TERDETEKSI JOY/ANGER:")
print("="*60)
# Tampilkan lagu Happy tapi AI tidak bilang Joy
odd_happy = happy_songs[happy_songs['AI_Thinks'] != 'joy']
if not odd_happy.empty:
    print(odd_happy[['Title', 'True_Mood', 'AI_Thinks', 'AI_Conf', 'Joy_Score']].to_string(index=False))
else:
    print("Semua lagu Happy terdeteksi Joy.")