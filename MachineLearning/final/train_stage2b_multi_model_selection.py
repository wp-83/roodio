import os
import glob
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow_hub as hub
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# --- MODELS ---
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
TARGET_COUNT = 40
FOLDS = 5
SEED = 43

print("🚀 EXP 44: MULTI-MODEL CHAMPION SELECTION")
print("   (Comparing 5 Architectures with Strict RFE Pipeline)")

# ================= 1. LOAD LABEL =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)]
    mood_map = dict(zip(df['id'], df['mood']))
    print(f"📊 Labels Loaded: {len(df)}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ================= 2. AUDIO ELITE SELECTION =================
print("\n⚖️ Running Audio Elite Selection...")
X_audio_raw, filenames = [], []

def get_id_smart(path):
    base = os.path.basename(path)
    return base.split('_')[0] if '_' in base else None

def extract_cluster_feat(path):
    y, sr = librosa.load(path, sr=16000)
    return [
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(np.std(librosa.feature.chroma_cens(y=y, sr=sr), axis=1))
    ]

all_files = []
for d in SOURCE_DIRS:
    all_files += glob.glob(os.path.join(d, '**/*.wav'), recursive=True)
    all_files += glob.glob(os.path.join(d, '**/*.mp3'), recursive=True)

for p in tqdm(all_files):
    fid = get_id_smart(p)
    if fid not in mood_map: continue
    try:
        X_audio_raw.append(extract_cluster_feat(p))
        filenames.append(p)
    except: pass

X_audio_raw = np.array(X_audio_raw)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_audio_raw)

kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=30)
clusters = kmeans.fit_predict(X_scaled)

bright0 = np.mean(X_audio_raw[clusters == 0, 0])
bright1 = np.mean(X_audio_raw[clusters == 1, 0])
sad_cluster, relaxed_cluster = (0, 1) if bright0 < bright1 else (1, 0)

dist = kmeans.transform(X_scaled)
sad = sorted([(i, dist[i][sad_cluster]) for i in range(len(filenames)) if clusters[i] == sad_cluster], key=lambda x: x[1])[:TARGET_COUNT]
relaxed = sorted([(i, dist[i][relaxed_cluster]) for i in range(len(filenames)) if clusters[i] == relaxed_cluster], key=lambda x: x[1])[:TARGET_COUNT]

selected_files = [filenames[i] for i, _ in sad + relaxed]
labels = np.array([0]*len(sad) + [1]*len(relaxed))
print(f"✅ Selected {len(selected_files)} elite audio samples.")

# ================= 3. FULL FEATURE EXTRACTION =================
print("\n⏳ Extracting Full Features...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_full(path):
    y, sr = librosa.load(path, sr=16000)
    y = np.pad(y, (0, max(0, 16000 - len(y))))
    _, emb, _ = yamnet(y)
    return np.concatenate([
        tf.reduce_mean(emb, axis=0).numpy(),
        np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1),
        [np.mean(librosa.feature.rms(y=y))]
    ])

X = np.array([extract_full(p) for p in tqdm(selected_files)])

# ================= 4. DEFINE MODELS =================
# Kita siapkan 5 kandidat model
models_config = {
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_split=5, random_state=SEED, n_jobs=-1),
    
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, min_samples_split=5, random_state=SEED, n_jobs=-1),
    
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=SEED),
    
    # SVM butuh scaling, jadi kita bungkus dalam pipeline nanti
    "SVM": SVC(C=1.0, kernel='rbf', probability=True, random_state=SEED),
    
    "LogisticRegression": LogisticRegression(C=0.5, solver='liblinear', random_state=SEED)
}

# ================= 5. MULTI-MODEL COMPETITION =================
print(f"\n🚀 STARTING 5-MODEL COMPETITION (5-Fold CV)")
print("-" * 60)
print(f"{'MODEL NAME':<20} | {'MEAN ACC':<10} | {'STD DEV':<10} | {'STATUS'}")
print("-" * 60)

results = []
best_score = 0
best_model_name = ""
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# RFE Base Selector (Menggunakan RF kecil untuk memilih fitur)
rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=SEED), n_features_to_select=30, step=0.05)

for name, model in models_config.items():
    accs = []
    
    # --- CROSS VALIDATION LOOP ---
    for tr, ts in skf.split(X, labels):
        X_tr, X_ts = X[tr], X[ts]
        y_tr, y_ts = labels[tr], labels[ts]
        
        # 1. Feature Selection (RFE) pada Train Data saja
        X_tr_rfe = rfe_selector.fit_transform(X_tr, y_tr)
        X_ts_rfe = rfe_selector.transform(X_ts)
        
        # 2. Scaling (Khusus SVM & LR)
        if name in ["SVM", "LogisticRegression"]:
            scaler_pipe = StandardScaler()
            X_tr_final = scaler_pipe.fit_transform(X_tr_rfe)
            X_ts_final = scaler_pipe.transform(X_ts_rfe)
        else:
            X_tr_final = X_tr_rfe
            X_ts_final = X_ts_rfe
            
        # 3. Train & Predict
        model.fit(X_tr_final, y_tr)
        pred = model.predict(X_ts_final)
        
        accs.append(accuracy_score(y_ts, pred))
    
    # --- STATS ---
    mean_acc = np.mean(accs) * 100
    std_acc = np.std(accs) * 100
    
    # Check Status
    status = ""
    if mean_acc > 80.0: status += "🌟 High Acc "
    if std_acc < 5.0:   status += "✅ Stable"
    
    print(f"{name:<20} | {mean_acc:.2f}%    | ±{std_acc:.2f}%   | {status}")
    
    results.append({
        'name': name,
        'mean': mean_acc,
        'std': std_acc,
        'model_obj': model,
        'needs_scaling': name in ["SVM", "LogisticRegression"]
    })

# ================= 6. SELECT CHAMPION =================
# Sort by Mean Accuracy Descending, then Std Ascending
results.sort(key=lambda x: (x['mean'], -x['std']), reverse=True)

champion = results[0]
print("=" * 60)
print(f"🏆 CHAMPION MODEL: {champion['name']}")
print(f"   Accuracy: {champion['mean']:.2f}% ± {champion['std']:.2f}%")
print("=" * 60)

# ================= 7. TRAIN FINAL MODEL & SAVE =================
print(f"\n💾 Training Final Model ({champion['name']}) on FULL DATA...")

# 1. RFE on Full Data
X_rfe_full = rfe_selector.fit_transform(X, labels)

# 2. Scaling on Full Data (If needed)
if champion['needs_scaling']:
    final_scaler = StandardScaler()
    X_final_input = final_scaler.fit_transform(X_rfe_full)
    joblib.dump(final_scaler, 'models/stage2b_scaler.pkl') # Save scaler
else:
    X_final_input = X_rfe_full

# 3. Train Classifier
final_clf = champion['model_obj']
final_clf.fit(X_final_input, labels)

# 4. Save
os.makedirs('models', exist_ok=True)
joblib.dump(rfe_selector, 'models/stage2b_rfe_selector.pkl') # Save Selector
joblib.dump(final_clf, 'models/stage2b_best_model.pkl') # Save Model

# Metadata text
with open('models/model_info.txt', 'w') as f:
    f.write(f"Model: {champion['name']}\n")
    f.write(f"Accuracy: {champion['mean']:.2f}%\n")
    f.write(f"Std Dev: {champion['std']:.2f}%\n")
    f.write(f"Needs Scaling: {champion['needs_scaling']}\n")

print(f"✅ Saved successfully as 'models/stage2b_best_model.pkl'")

# ================= 8. CONFUSION MATRIX (CHAMPION) =================
# Visualize the champion's performance using cross_val_predict logic
from sklearn.model_selection import cross_val_predict

print("\n📊 Generating Champion Confusion Matrix...")
# Pipeline Re-construction for Visualization
viz_steps = [('rfe', rfe_selector)]
if champion['needs_scaling']:
    viz_steps.append(('scaler', StandardScaler()))
viz_steps.append(('clf', champion['model_obj']))

viz_pipeline = Pipeline(viz_steps)
y_pred_viz = cross_val_predict(viz_pipeline, X, labels, cv=skf)

print(classification_report(labels, y_pred_viz, target_names=['sad', 'relaxed']))

cm = confusion_matrix(labels, y_pred_viz)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sad', 'Relaxed'],
            yticklabels=['Sad', 'Relaxed'])
plt.title(f"{champion['name']} Result\nAcc: {champion['mean']:.1f}%")
plt.show()