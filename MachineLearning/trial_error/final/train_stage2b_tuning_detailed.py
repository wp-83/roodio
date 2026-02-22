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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
FOLDS = 6
SEED = 43

print("🚀 EXP 46: TUNING CHAMPION (WITH DETAILED CV & STD)")
print("   (Mencari Model Paling Stabil di Atas 80%)")

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

# ================= 4. DEFINE PARAMETER GRIDS =================
param_grids = {
    "RandomForest": {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2, 5]
    },
    "ExtraTrees": {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2, 5]
    },
    "GradientBoosting": {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1, 0.2],
        'model__max_depth': [3, 5]
    },
    "SVM": {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto'],
        'model__kernel': ['rbf', 'linear']
    },
    "LogisticRegression": {
        'model__C': [0.1, 1, 10],
        'model__solver': ['liblinear']
    }
}

base_models = {
    "RandomForest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    "SVM": SVC(probability=True, random_state=SEED),
    "LogisticRegression": LogisticRegression(random_state=SEED)
}

# ================= 5. TUNING & DETAILED REPORTING =================
print(f"\n🚀 STARTING HYPERPARAMETER TUNING (GridSearch with Detailed Stats)...")
print("=" * 115)
print(f"{'MODEL':<20} | {'MEAN ACC':<10} | {'STD DEV':<10} | {'FOLD 1':<7} | {'FOLD 2':<7} | {'FOLD 3':<7} | {'FOLD 4':<7} | {'FOLD 5':<7}")
print("-" * 115)

results = []
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=SEED), n_features_to_select=30, step=0.05)

for name, model in base_models.items():
    
    # 1. Build Pipeline
    steps = [('selection', rfe)]
    if name in ["SVM", "LogisticRegression"]:
        steps.insert(0, ('scaler', StandardScaler()))
    steps.append(('model', model))
    pipeline = Pipeline(steps)
    
    # 2. Grid Search
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X, labels)
    
    # 3. EXTRACT DETAILED STATS (INI BAGIAN BARUNYA)
    best_idx = grid.best_index_
    cv_res = grid.cv_results_
    
    mean_score = cv_res['mean_test_score'][best_idx] * 100
    std_score  = cv_res['std_test_score'][best_idx] * 100
    
    # Ambil nilai per fold
    folds = []
    for i in range(FOLDS):
        folds.append(cv_res[f'split{i}_test_score'][best_idx] * 100)
    
    # Print Table Row
    fold_str = " | ".join([f"{f:.1f}%" for f in folds])
    print(f"{name:<20} | {mean_score:.2f}%    | ±{std_score:.2f}%   | {fold_str}")
    
    results.append({
        'name': name,
        'accuracy': mean_score,
        'std': std_score,
        'best_estimator': grid.best_estimator_,
        'best_params': grid.best_params_
    })

# ================= 6. SELECT CHAMPION =================
# Sort by Mean Accuracy Descending, then Std Ascending (Lower is better)
results.sort(key=lambda x: (x['accuracy'], -x['std']), reverse=True)
champion = results[0]

print("=" * 115)
print(f"🏆 ULTIMATE CHAMPION: {champion['name']}")
print(f"   Accuracy: {champion['accuracy']:.2f}% (Std: ±{champion['std']:.2f}%)")
print(f"   Params:   {champion['best_params']}")
print("=" * 115)

# ================= 7. SAVE FINAL MODEL =================
print(f"\n💾 Saving Final Tuned Model...")
final_model = champion['best_estimator']
os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/stage2b_tuned_champion.pkl')
print(f"✅ Saved to: models/stage2b_tuned_champion.pkl")

# ================= 8. VISUALIZE CHAMPION =================
print("\n📊 Generating Confusion Matrix (Champion)...")
y_pred_viz = cross_val_predict(final_model, X, labels, cv=skf)

print(classification_report(labels, y_pred_viz, target_names=['sad', 'relaxed']))

cm = confusion_matrix(labels, y_pred_viz)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sad', 'Relaxed'],
            yticklabels=['Sad', 'Relaxed'])
plt.title(f"{champion['name']} (Tuned)\nAcc: {champion['accuracy']:.1f}% ±{champion['std']:.1f}%")
plt.show()