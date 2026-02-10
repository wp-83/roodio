import os
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import re

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']

print("🚀 OPTIMASI THRESHOLD MANUAL (STAGE 2B)")
print("   Mencari aturan IF-ELSE terbaik untuk memisahkan Sad vs Relaxed.")

# ================= 1. PREPARE DATA =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    lyrics_map = dict(zip(df['id'], df['lyrics']))
    mood_map = dict(zip(df['id'], df['mood']))
    for k, v in lyrics_map.items(): 
        if pd.isna(v): lyrics_map[k] = ""
except: exit()

def get_id(path):
    base = os.path.basename(path)
    return base.split('_')[0] if '_' in base else None

# Collect Files
files = []
labels = [] # 0=Sad, 1=Relaxed
lyrics_list = []

for d in SOURCE_DIRS:
    for m in TARGET_MOODS:
        fs = glob.glob(os.path.join(d, m, '*.wav')) + glob.glob(os.path.join(d, m, '*.mp3'))
        for f in fs:
            fid = get_id(f)
            if fid in lyrics_map:
                files.append(f)
                labels.append(0 if m == 'sad' else 1)
                lyrics_list.append(lyrics_map[fid])

print(f"📦 Total Data: {len(files)} (Sad={labels.count(0)}, Relaxed={labels.count(1)})")

# ================= 2. CALCULATE SCORES =================
print("\n⏳ Calculating Sentiment & Audio Scores...")
nlp = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

data_scores = []

for i in tqdm(range(len(files))):
    # 1. Text Score
    text = re.sub(r"[^a-z0-9\s]", '', str(lyrics_list[i]).lower())[:512]
    if not text: 
        s_sad, s_joy_neu = 0, 0
    else:
        try:
            res = nlp(text)[0]
            sc = {item['label']: item['score'] for item in res}
            # Skor Sadness Murni
            s_sad = sc.get('sadness', 0) + sc.get('fear', 0)*0.5 # Fear sering muncul di sad
            # Skor Relaxed (Joy + Neutral)
            s_joy_neu = sc.get('joy', 0) + sc.get('neutral', 0)
        except: s_sad, s_joy_neu = 0, 0
        
    # 2. Audio Score (RMS - Volume)
    try:
        y, _ = librosa.load(files[i], sr=16000)
        rms = np.mean(librosa.feature.rms(y=y))
    except: rms = 0
    
    data_scores.append({
        'sad_score': s_sad,
        'relax_score': s_joy_neu,
        'rms': rms,
        'true_label': labels[i]
    })

df_res = pd.DataFrame(data_scores)

# ================= 3. BRUTE FORCE OPTIMIZATION =================
print("\n🔍 Searching for the Golden Threshold...")

best_acc = 0
best_rule = ""
best_params = {}

# Skenario 1: Hanya Sadness Score (Text)
# Aturan: Jika Sadness > X, maka SAD. Else RELAXED.
thresholds = np.linspace(0, 1, 100)
for t in thresholds:
    # Prediksi: 0(Sad) jika score > t, else 1(Relaxed)
    preds = [0 if x > t else 1 for x in df_res['sad_score']]
    acc = accuracy_score(df_res['true_label'], preds)
    
    if acc > best_acc:
        best_acc = acc
        best_rule = "Sadness > Threshold"
        best_params = {'t': t}

# Skenario 2: Sadness Ratio (Text)
# Aturan: Jika Sadness / (Sadness + RelaxScore) > X, maka SAD.
for t in thresholds:
    preds = []
    for _, row in df_res.iterrows():
        total = row['sad_score'] + row['relax_score'] + 1e-9
        ratio = row['sad_score'] / total
        preds.append(0 if ratio > t else 1)
        
    acc = accuracy_score(df_res['true_label'], preds)
    if acc > best_acc:
        best_acc = acc
        best_rule = "Sadness Ratio > Threshold"
        best_params = {'t': t}

# Skenario 3: Hybrid (Text + Audio)
# Aturan: Jika Sadness > T1 OR RMS < T2 (Lagu sangat pelan), maka SAD.
rms_thresholds = np.linspace(df_res['rms'].min(), df_res['rms'].max(), 50)
for t1 in thresholds: # Sadness Threshold
    for t2 in rms_thresholds: # RMS Threshold
        preds = []
        for _, row in df_res.iterrows():
            # Logika: Jika lirik sedih ATAU lagu sangat pelan -> SAD
            is_sad = (row['sad_score'] > t1) or (row['rms'] < t2)
            preds.append(0 if is_sad else 1)
            
        acc = accuracy_score(df_res['true_label'], preds)
        if acc > best_acc:
            best_acc = acc
            best_rule = "Hybrid (Sadness > T1 OR RMS < T2)"
            best_params = {'t_sad': t1, 't_rms': t2}

# ================= 4. REPORT =================
print("\n" + "="*50)
print(f"🏆 HASIL OPTIMASI TERBAIK")
print("="*50)
print(f"✅ Akurasi Maksimal : {best_acc*100:.2f}%")
print(f"📜 Aturan Terbaik   : {best_rule}")
print(f"⚙️ Parameter        : {best_params}")

print("\n--- Analisis Prediksi dengan Aturan Terbaik ---")
# Generate Final Preds for Report
final_preds = []
if "Ratio" in best_rule:
    t = best_params['t']
    for _, row in df_res.iterrows():
        ratio = row['sad_score'] / (row['sad_score'] + row['relax_score'] + 1e-9)
        final_preds.append(0 if ratio > t else 1)
elif "Hybrid" in best_rule:
    t1, t2 = best_params['t_sad'], best_params['t_rms']
    for _, row in df_res.iterrows():
        is_sad = (row['sad_score'] > t1) or (row['rms'] < t2)
        final_preds.append(0 if is_sad else 1)
else: # Sadness Only
    t = best_params['t']
    final_preds.append([0 if x > t else 1 for x in df_res['sad_score']])

# Handle shape mismatch if simplistic loop used
if len(final_preds) != len(labels): # Re-run simple logic carefully
    final_preds = [0 if x > best_params['t'] else 1 for x in df_res['sad_score']]

print(classification_report(labels, final_preds, target_names=['Sad', 'Relaxed']))

# Simpan Aturan ini ke file teks agar bisa dibaca script Inference nanti
import json
with open('models/stage2b_threshold_rules.json', 'w') as f:
    json.dump({
        'rule': best_rule,
        'params': best_params,
        'accuracy': best_acc
    }, f)
print("✅ Aturan disimpan ke 'models/stage2b_threshold_rules.json'")