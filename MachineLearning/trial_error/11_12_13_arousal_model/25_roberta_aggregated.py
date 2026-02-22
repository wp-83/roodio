import os
import logging

# Mute Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- KONFIGURASI ---
LYRICS_PATH = 'data/lyrics/lyrics.csv'
TARGET_MOODS = ['angry', 'happy'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

print(f"🚀 MEMULAI EXP 25: AGGREGATED RoBERTa (TEAM BATTLE)...")

# --- 1. LOAD DATA ---
if not os.path.exists(LYRICS_PATH):
    print("❌ File CSV tidak ditemukan.")
    exit()

try:
    df = pd.read_csv(LYRICS_PATH, sep=';')
    if len(df.columns) == 1: df = pd.read_csv(LYRICS_PATH, sep=',')
    df.columns = df.columns.str.strip().str.lower()
except: exit()

df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()
print(f"📋 Total Data: {len(df)}")

# --- 2. LOAD MODEL ---
classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 3. INFERENCE LOOP (TEAM BATTLE) ---
y_true = []
y_pred = []
happy_scores_log = []
angry_scores_log = []
titles_log = []

print("🧠 Menghitung Skor Tim Happy vs Tim Angry...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    lyrics = str(row['lyrics'])
    true_mood = row['mood']
    title = row['title'] if 'title' in df.columns else f"Song {index}"
    
    if len(lyrics) < 2: continue

    try:
        # 1. Dapatkan Raw Scores (Semua Emosi)
        output = classifier(lyrics)[0] 
        # output = [{'label': 'joy', 'score': 0.1}, {'label': 'neutral', 'score': 0.8}...]
        
        # 2. Convert ke Dictionary biar gampang panggil
        scores = {item['label']: item['score'] for item in output}
        
        # 3. HITUNG SKOR TIM (AGGREGATION)
        # Team Happy (Positive vibe)
        score_team_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        
        # Team Angry (Negative vibe)
        score_team_angry = (scores.get('anger', 0) + 
                            scores.get('disgust', 0) + 
                            scores.get('fear', 0) + 
                            scores.get('sadness', 0))
        
        # Neutral diabaikan (dianggap noise)
        
        # 4. TENTUKAN PEMENANG
        if score_team_happy > score_team_angry:
            pred = 'happy'
        else:
            pred = 'angry'
            
        y_true.append(true_mood)
        y_pred.append(pred)
        
        # Log untuk analisis
        happy_scores_log.append(score_team_happy)
        angry_scores_log.append(score_team_angry)
        titles_log.append(title)
        
    except Exception as e:
        print(f"Error {title}: {e}")

# --- 4. EVALUASI ---
print("\n" + "="*50)
print("📊 HASIL AKHIR (AGGREGATED STRATEGY)")
print("="*50)

acc = accuracy_score(y_true, y_pred) * 100
print(f"🏆 Accuracy: {acc:.2f}%")

print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['angry', 'happy']))

# Matrix
labels = ['angry', 'happy']
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
plt.title(f'Aggregated Lirik Analysis\nAccuracy: {acc:.2f}%')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig('cm_exp25_aggregated.png')
plt.show()

# --- 5. ANALISIS ERROR ---
df_res = pd.DataFrame({
    'Title': titles_log, 
    'True': y_true, 
    'Pred': y_pred, 
    'Team_Happy_Score': happy_scores_log, 
    'Team_Angry_Score': angry_scores_log
})

errors = df_res[df_res['True'] != df_res['Pred']]

if len(errors) > 0:
    print(f"\n❌ MASIH SALAH TEBAK ({len(errors)} lagu):")
    print("-" * 75)
    print(f"{'Title':<25} | {'True':<8} | {'Pred':<8} | {'HappyScr':<8} | {'AngryScr':<8}")
    print("-" * 75)
    for _, row in errors.iterrows():
        print(f"{str(row['Title'])[:25]:<25} | {row['True']:<8} | {row['Pred']:<8} | {row['Team_Happy_Score']:.3f}    | {row['Team_Angry_Score']:.3f}")
else:
    print("\n🎉 SEMPURNA! Strategi Tim Berhasil 100%.")