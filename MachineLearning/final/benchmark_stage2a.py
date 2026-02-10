import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline

# --- MODELS & TUNING ---
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['angry', 'happy'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI TUNED BENCHMARK (GRID SEARCH)...")

# ================= 1. PREPARE DATA (SAMA SEPERTI SEBELUMNYA) =================
# ... (Bagian loading data ini sama persis agar adil) ...
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    lyrics_map = pd.Series(df.lyrics.values, index=df.id).to_dict()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Loaded: {len(df)}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Text Cleaning
def clean_lyrics_text(text):
    if pd.isna(text) or text == '': return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text) 
    text = re.sub(r'\(.*?\)', ' ', text) 
    garbage = ['lyrics', 'embed', 'contributors', 'translation']
    for w in garbage: text = text.replace(w, '')
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for pid in lyrics_map: lyrics_map[pid] = clean_lyrics_text(lyrics_map[pid])

print("⏳ Loading Extractors...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

X_combined = []
y_labels = []

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
        _, emb, _ = yamnet_model(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, [rms, zcr]])
    except: return None

def get_text_scores(lyrics):
    try:
        output = nlp_classifier(str(lyrics)[:512])[0]
        scores = {item['label']: item['score'] for item in output}
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
        return [s_angry, s_happy] 
    except: return [0.5, 0.5]

def get_id(path):
    base = os.path.basename(path)
    return base.split('_')[0].strip() if '_' in base else None

# Feature Extraction Loop
all_files = []
for d in SOURCE_DIRS:
    all_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

print("🧠 Extracting Features...")
for path in tqdm(all_files):
    fid = get_id(path)
    if fid not in lyrics_map: continue
    mood = mood_map[fid]
    label = 0 if mood == 'angry' else 1
    
    aud = extract_audio(path)
    txt = get_text_scores(lyrics_map[fid])
    
    if aud is not None:
        X_combined.append(np.concatenate([aud, txt]))
        y_labels.append(label)

X = np.array(X_combined)
y = np.array(y_labels)

# ================= 2. DEFINISI GRID SEARCH =================
print(f"\n🚀 STARTING HYPERPARAMETER TUNING...")
print(f"   (Mencari setting terbaik untuk setiap model klasik)")

# Kita bungkus model dalam Pipeline agar Scaling otomatis dilakukan
# Struktur: Pipeline([('scaler', StandardScaler()), ('clf', Model())])

# 1. Random Forest (Biasanya tidak butuh scaling, tapi kita masukkan saja biar seragam)
pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=SEED))])
grid_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}

# 2. SVM (Wajib Scaling)
pipe_svm = Pipeline([('scaler', StandardScaler()), ('clf', SVC(random_state=SEED))])
grid_svm = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 'auto'],
    'clf__kernel': ['rbf', 'linear'] # Coba Linear juga
}

# 3. Logistic Regression
pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=SEED, max_iter=1000))])
grid_lr = {
    'clf__C': [0.1, 1, 10],
    'clf__solver': ['liblinear', 'lbfgs']
}

# 4. Gradient Boosting
pipe_gb = Pipeline([('scaler', StandardScaler()), ('clf', GradientBoostingClassifier(random_state=SEED))])
grid_gb = {
    'clf__n_estimators': [100, 200],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__max_depth': [3, 5]
}

# 5. KNN
pipe_knn = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])
grid_knn = {
    'clf__n_neighbors': [3, 5, 7, 9],
    'clf__weights': ['uniform', 'distance']
}

# Gabungkan dalam Dictionary untuk Loop
models_to_tune = {
    "Random Forest": (pipe_rf, grid_rf),
    "SVM": (pipe_svm, grid_svm),
    "Logistic Regression": (pipe_lr, grid_lr),
    "Gradient Boosting": (pipe_gb, grid_gb),
    "KNN": (pipe_knn, grid_knn)
}

# ================= 3. TUNING LOOP =================
results_tuned = {}
best_params_log = {}

print("-" * 80)
print(f"{'MODEL':<20} | {'TUNED ACC':<10} | {'STD DEV':<10} | {'BEST PARAMS'}")
print("-" * 80)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for name, (pipeline_obj, params) in models_to_tune.items():
    # Grid Search CV
    grid = GridSearchCV(pipeline_obj, params, cv=skf, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X, y)
    
    # Ambil Hasil Terbaik
    best_acc = grid.best_score_ * 100
    best_std = grid.cv_results_['std_test_score'][grid.best_index_] * 100
    
    results_tuned[name] = {
        'mean': best_acc, 
        'std': best_std,
        'scores_per_fold': [grid.cv_results_[f'split{i}_test_score'][grid.best_index_] for i in range(FOLDS)]
    }
    
    # Format Params String agar tidak kepanjangan
    param_str = str(grid.best_params_).replace("clf__", "")
    best_params_log[name] = param_str
    
    print(f"{name:<20} | {best_acc:.2f}%    | ±{best_std:.2f}%   | {param_str[:40]}...")

# --- Masukkan Hasil Stacking Kamu (Manual) ---
# Update nilai ini sesuai hasil 'train_stage2a_stacking.py' kamu
STACKING_ACC = 84.00 
STACKING_STD = 4.00
results_tuned["YOUR STACKING"] = {'mean': STACKING_ACC, 'std': STACKING_STD}

print("-" * 80)

# ================= 4. VISUALISASI =================
plt.figure(figsize=(12, 6))

names = list(results_tuned.keys())
means = [results_tuned[n]['mean'] for n in names]
stds = [results_tuned[n]['std'] for n in names]

# Bar Plot dengan Error Bar (Std Dev)
bars = plt.bar(names, means, yerr=stds, capsize=5, color=['#d1e7dd' if 'STACKING' not in n else '#198754' for n in names], edgecolor='black')

plt.title('Tuned Classic Models vs Stacking Ensemble (Angry/Happy)')
plt.ylabel('Accuracy (%)')
plt.ylim(60, 100) # Zoom in ke area 60-100% biar bedanya kelihatan
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Tambahkan label angka di atas bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_tuned_result.png')
plt.show()

print("\n✅ Tuned Benchmark Selesai!")
print("   Lihat 'benchmark_tuned_result.png' untuk grafik perbandingan.")
print("   Lihat Log di atas untuk parameter terbaik setiap model.")