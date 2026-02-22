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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
TARGET_COUNT = 40  # Total 80 Data
FOLDS = 5 # Kita gunakan 5 Fold standar
SEED = 43

print("🚀 EXP 48: EXTENSIVE HYPERPARAMETER TUNING")
print("   Goal: Maximize Accuracy AND Minimize Std Dev")

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
print("\n⚖️ Re-creating Elite Dataset (Top 40)...")
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

for p in tqdm(all_files, desc="Scanning & Clustering"):
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
print("\n⏳ Extracting Full Features (YAMNet + Chroma)...")
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

# ================= 4. DEFINE PIPELINE & PARAM GRID =================
print("\n🔨 Configuring Extensive Grid Search...")

# Base Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), # 1. Scaling
    ('selection', RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=SEED))), # 2. Feature Selection
    ('model', SVC(probability=True, random_state=SEED)) # 3. Classifier
])

# PARAMETER GRID (MENU YANG AKAN DICOBA)
# Kita perluas jangkauannya untuk mencari stabilitas
param_grid = [
    # Skenario 1: Kernel Linear (Biasanya stabil di data sedikit)
    {
        'selection__n_features_to_select': [15, 20, 25, 30, 40, 50], # Coba jumlah fitur sedikit s.d banyak
        'model__kernel': ['linear'],
        'model__C': [0.01, 0.05, 0.1, 0.5, 1, 5] # Coba margin lebar (0.01) s.d ketat (5)
    },
    # Skenario 2: Kernel RBF (Untuk menangkap pola non-linear)
    {
        'selection__n_features_to_select': [15, 20, 25, 30],
        'model__kernel': ['rbf'],
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto', 0.1, 0.01]
    }
]

# ================= 5. RUN GRID SEARCH =================
print(f"🚀 Starting Grid Search (This might take a while)...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)

grid.fit(X, labels)

# ================= 6. ANALYZE RESULTS (FOCUS ON STD) =================
print("\n" + "="*80)
print(f"📊 TOP 5 CONFIGURATIONS (Sorted by Accuracy, then Low Std)")
print("="*80)
print(f"{'RANK':<5} | {'MEAN ACC':<10} | {'STD DEV':<10} | {'PARAMS'}")
print("-" * 80)

# Ambil hasil
results_df = pd.DataFrame(grid.cv_results_)
# Urutkan berdasarkan Mean Score (Desc) lalu Std Score (Asc)
results_df = results_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=[False, True])

for i in range(min(5, len(results_df))):
    row = results_df.iloc[i]
    mean = row['mean_test_score'] * 100
    std = row['std_test_score'] * 100
    params = str(row['params'])
    print(f"{i+1:<5} | {mean:.2f}%    | ±{std:.2f}%   | {params[:90]}...")

# Ambil Model Terbaik
best_model = grid.best_estimator_
best_acc = grid.best_score_ * 100
best_std = results_df.iloc[0]['std_test_score'] * 100

print("\n" + "="*50)
print(f"🏆 CHAMPION SELECTED")
print("="*50)
print(f"✅ Accuracy : {best_acc:.2f}%")
print(f"📉 Std Dev  : ±{best_std:.2f}%")
print(f"⚙️ Params   : {grid.best_params_}")

# ================= 7. VISUALIZE CHAMPION =================
print("\n📊 Generating Confusion Matrix (Champion)...")
y_pred_cv = cross_val_predict(best_model, X, labels, cv=skf)

print(classification_report(labels, y_pred_cv, target_names=['sad', 'relaxed']))

cm = confusion_matrix(labels, y_pred_cv)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sad', 'Relaxed'], 
            yticklabels=['Sad', 'Relaxed'])
plt.title(f"Tuned Champion\nAcc: {best_acc:.2f}% ±{best_std:.2f}%")
plt.show()

# ================= 8. SAVE MODEL =================
print("\n💾 Saving Best Model...")

# Fit ulang ke seluruh data
best_model.fit(X, labels)

if not os.path.exists('models'): os.makedirs('models')
joblib.dump(best_model, 'models/stage2b_tuned_final.pkl')

# Save Metadata
with open('models/stage2b_info.txt', 'w') as f:
    f.write(f"Model: SVM Tuned Pipeline\n")
    f.write(f"Accuracy: {best_acc:.2f}%\n")
    f.write(f"Std Dev: {best_std:.2f}%\n")
    f.write(f"Best Params: {grid.best_params_}\n")

print("✅ DONE! Model saved to: models/stage2b_tuned_final.pkl")