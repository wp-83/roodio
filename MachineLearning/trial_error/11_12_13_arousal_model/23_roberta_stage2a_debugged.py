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

print(f"🚀 MEMULAI EXP 23 (DEBUGGED): PRETRAINED RoBERTa...")

# --- 1. LOAD DATA ---
if not os.path.exists(LYRICS_PATH):
    print(f"❌ Error: File {LYRICS_PATH} tidak ditemukan!")
    exit()

try:
    df = pd.read_csv(LYRICS_PATH, sep=';')
    df.columns = df.columns.str.strip().str.lower()
    
    # Fallback jika cuma 1 kolom terdeteksi (berarti delimiter salah)
    if len(df.columns) == 1:
        print("⚠️ Separator ';' sepertinya salah, mencoba separator ','...")
        df = pd.read_csv(LYRICS_PATH, sep=',')
        df.columns = df.columns.str.strip().str.lower()

    # Cek kolom wajib
    if 'lyrics' not in df.columns or 'mood' not in df.columns:
        print(f"❌ Kolom Wajib (lyrics, mood) tidak ada. Kolom ditemukan: {df.columns.tolist()}")
        exit()
        
except Exception as e:
    print(f"❌ Gagal membaca CSV: {e}")
    exit()

# Filter Data
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)].copy()
print(f"📋 Total Data (Angry + Happy): {len(df)}")

# --- 2. LOAD MODEL (DENGAN PARAMETER BARU) ---
print(f"⏳ Loading Model NLP...")
# PERBAIKAN: Ganti return_all_scores=True menjadi top_k=None
classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)
print("✅ Model Siap!")

# --- 3. INFERENCE LOOP ---
y_true = []
y_pred = []
probas_joy = []
probas_anger = []
titles_log = [] 

print("🧠 Sedang Menganalisis...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    lyrics = str(row['lyrics'])
    true_label = row['mood']
    title = row['title'] if 'title' in df.columns else f"Song {index}"
    
    # Skip jika lirik kosong/NaN
    if len(lyrics.strip()) < 2 or lyrics.lower() == 'nan':
        continue

    try:
        # Inference
        raw_output = classifier(lyrics)
        
        # PERBAIKAN LOGIKA PARSING
        # Output top_k=None biasanya: [[{'label': 'joy', 'score': 0.9}, {'label': 'anger'...}]]
        # Kita ambil list pertama
        results = raw_output[0]
        
        # Debugging baris pertama saja biar user tau formatnya
        if index == 0:
            print(f"\n🔎 DEBUG FORMAT OUTPUT (Lagu Pertama): {results[:2]}...") 
        
        # Cari skor Joy dan Anger
        score_joy = 0.0
        score_anger = 0.0
        
        for item in results:
            if item['label'] == 'joy':
                score_joy = item['score']
            elif item['label'] == 'anger':
                score_anger = item['score']
        
        # Duel Skor
        if score_joy > score_anger:
            predicted_label = 'happy'
        else:
            predicted_label = 'angry'
            
        y_true.append(true_label)
        y_pred.append(predicted_label)
        probas_joy.append(score_joy)
        probas_anger.append(score_anger)
        titles_log.append(title)
        
    except Exception as e:
        print(f"⚠️ Error pada '{title}': {e}")
        # Print format raw jika error untuk diagnosa
        try: print(f"   Raw Output saat error: {raw_output}")
        except: pass

# --- 4. EVALUASI ---
if len(y_true) == 0:
    print("\n❌ TIDAK ADA DATA YANG BERHASIL DI PROSES.")
    print("   Cek apakah kolom 'lyrics' di CSV benar-benar ada isinya?")
    exit()

print("\n" + "="*50)
print("📊 HASIL AKHIR (RoBERTa)")
print("="*50)

acc = accuracy_score(y_true, y_pred) * 100
print(f"🏆 Accuracy: {acc:.2f}%")

print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['angry', 'happy']))

# Matrix
labels = ['angry', 'happy']
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=labels, yticklabels=labels)
plt.title(f'Lirik Sentiment Analysis\nAccuracy: {acc:.2f}%')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.savefig('cm_exp23_debugged.png')
plt.show()

# --- 5. ANALISIS ERROR ---
df_res = pd.DataFrame({'Title': titles_log, 'True': y_true, 'Pred': y_pred, 'Joy': probas_joy, 'Anger': probas_anger})
errors = df_res[df_res['True'] != df_res['Pred']]

if len(errors) > 0:
    print(f"\n❌ SALAH TEBAK ({len(errors)} lagu):")
    print("-" * 60)
    print(f"{'Title':<25} | {'True':<8} | {'Pred':<8} | {'Joy':<5} | {'Anger':<5}")
    print("-" * 60)
    for _, row in errors.iterrows():
        print(f"{str(row['Title'])[:25]:<25} | {row['True']:<8} | {row['Pred']:<8} | {row['Joy']:.2f}  | {row['Anger']:.2f}")
else:
    print("\n🎉 SEMPURNA! Tidak ada kesalahan.")