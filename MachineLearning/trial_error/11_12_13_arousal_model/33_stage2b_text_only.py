import os
import logging

# --- 0. ENVIRONMENT SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. CONFIGURATION ---
LYRICS_PATH = 'data/lyrics/lyrics_cleaned.csv'
TARGET_MOODS = ['sad', 'relaxed'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 33: STAGE 2B TEXT ONLY (ROBERTA FEATURES)...")

# --- 2. DATA LOADING ---
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
print(f"   Sad: {len(df[df['mood']=='sad'])}")
print(f"   Relaxed: {len(df[df['mood']=='relaxed'])}")

# --- 3. MODEL LOADING ---
print("⏳ Loading RoBERTa...")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 4. FEATURE EXTRACTION (TEXT ONLY) ---
X_features = [] 
y_labels = []

# List nama fitur untuk analisis nanti
feature_names = ['joy', 'neutral', 'surprise', 'sadness', 'fear', 'anger']

print("🧠 Extracting Emotion Scores from Lyrics...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row['lyrics'])
    label = 0 if row['mood'] == 'sad' else 1 # 0=Sad, 1=Relaxed
    
    try:
        if len(text) < 2:
            scores = {k: 0.0 for k in feature_names}
        else:
            output = nlp_classifier(text)[0]
            scores = {item['label']: item['score'] for item in output}
        
        # Urutan harus konsisten!
        feat_vec = [
            scores.get('joy', 0),
            scores.get('neutral', 0),
            scores.get('surprise', 0),
            scores.get('sadness', 0),
            scores.get('fear', 0),
            scores.get('anger', 0)
        ]
        
        X_features.append(feat_vec)
        y_labels.append(label)
        
    except Exception as e:
        print(f"⚠️ Error: {e}")

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {X_features.shape}")

# --- 5. TRAINING (RANDOM FOREST) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
y_true_all = []
y_pred_all = []
feature_importances_log = []

print(f"\n🚀 START TRAINING...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_labels)):
    X_tr, X_ts = X_features[train_idx], X_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_ts)
    
    # Log Importance
    feature_importances_log.append(clf.feature_importances_)
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred)
    
    acc = accuracy_score(y_ts, y_pred)
    print(f"   Fold {fold+1}: {acc*100:.0f}%")

# --- 6. REPORT ---
print("\n" + "="*50)
print("📊 HASIL AKHIR TEXT ONLY (STAGE 2B)")
print("="*50)

final_acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"🏆 Accuracy: {final_acc:.2f}%")
print(classification_report(y_true_all, y_pred_all, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Sad','Relaxed'], yticklabels=['Sad','Relaxed'])
plt.title(f'Text-Only Result\nAccuracy: {final_acc:.2f}%')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.savefig('cm_exp33_text_only.png')
plt.show()

# --- 7. EMOTION IMPORTANCE ANALYSIS ---
# Emosi apa yang paling berguna untuk membedakan Sad vs Relaxed?
avg_imp = np.mean(feature_importances_log, axis=0)
indices = np.argsort(avg_imp)[::-1] # Sort descending

print("\n🔑 KEYWORDS IMPORTANCE (Emosi RoBERTa mana yang paling berpengaruh?):")
print("-" * 60)
for f in range(X_features.shape[1]):
    idx = indices[f]
    print(f"{f+1}. {feature_names[idx].upper():<10} : {avg_imp[idx]:.4f}")
print("-" * 60)