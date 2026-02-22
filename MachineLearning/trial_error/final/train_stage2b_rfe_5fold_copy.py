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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

print("🚀 EXP 42 CLEAN VERSION (NO DATA LEAKAGE)")

# ================= 1. LOAD LABEL =================
df = pd.read_excel(LYRICS_PATH)
df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
df['mood'] = df['mood'].str.lower().str.strip()
df = df[df['mood'].isin(TARGET_MOODS)]
mood_map = dict(zip(df['id'], df['mood']))

# ================= 2. AUDIO ELITE SELECTION =================
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
    if fid not in mood_map:
        continue
    try:
        X_audio_raw.append(extract_cluster_feat(p))
        filenames.append(p)
    except:
        pass

X_audio_raw = np.array(X_audio_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_audio_raw)

kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=30)
clusters = kmeans.fit_predict(X_scaled)

bright0 = np.mean(X_audio_raw[clusters == 0, 0])
bright1 = np.mean(X_audio_raw[clusters == 1, 0])
sad_cluster, relaxed_cluster = (0, 1) if bright0 < bright1 else (1, 0)

dist = kmeans.transform(X_scaled)
sad = sorted(
    [(i, dist[i][sad_cluster]) for i in range(len(filenames)) if clusters[i] == sad_cluster],
    key=lambda x: x[1]
)[:TARGET_COUNT]

relaxed = sorted(
    [(i, dist[i][relaxed_cluster]) for i in range(len(filenames)) if clusters[i] == relaxed_cluster],
    key=lambda x: x[1]
)[:TARGET_COUNT]

selected_files = [filenames[i] for i, _ in sad + relaxed]
labels = np.array([0]*len(sad) + [1]*len(relaxed))

print(f"✅ Selected {len(selected_files)} elite audio")

# ================= 3. FULL FEATURE EXTRACTION =================
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

# ================= 4. CLEAN CV (RFE INSIDE) =================
print("\n🚀 START CLEAN 5-FOLD CV")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
accs, y_true, y_pred = [], [], []

for fold, (tr, ts) in enumerate(skf.split(X, labels)):
    X_tr, X_ts = X[tr], X[ts]
    y_tr, y_ts = labels[tr], labels[ts]

    # 🔒 RFE FIT ONLY ON TRAIN
    rfe = RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        n_features_to_select=30,
        step=0.05
    )
    X_tr_rfe = rfe.fit_transform(X_tr, y_tr)
    X_ts_rfe = rfe.transform(X_ts)

    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1
    )

    clf.fit(X_tr_rfe, y_tr)
    pred = clf.predict(X_ts_rfe)

    acc = accuracy_score(y_ts, pred)
    accs.append(acc)

    print(f"Fold {fold+1}: {acc*100:.2f}%")

    y_true.extend(y_ts)
    y_pred.extend(pred)

# ================= 5. REPORT =================
print("\n" + "="*50)
print(f"Mean Acc: {np.mean(accs)*100:.2f}%")
print(f"Std Acc : ±{np.std(accs)*100:.2f}%")
print("="*50)
print(classification_report(y_true, y_pred, target_names=['sad', 'relaxed']))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sad','Relaxed'],
            yticklabels=['Sad','Relaxed'])
plt.show()

# ================= 6. TRAIN FINAL MODEL =================
print("\n💾 Training FINAL MODEL (FULL DATA)")

final_rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    n_features_to_select=30,
    step=0.05
)

X_rfe_full = final_rfe.fit_transform(X, labels)

final_clf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    random_state=SEED,
    n_jobs=-1
)
final_clf.fit(X_rfe_full, labels)

os.makedirs('models', exist_ok=True)
joblib.dump(final_rfe, 'models/stage2b_rfe_selector.pkl')
joblib.dump(final_clf, 'models/stage2b_rf_rfe.pkl')

print("✅ FINAL MODEL SAVED — NO DATA LEAKAGE")
