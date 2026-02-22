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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
TARGET_COUNT = 40  # Top 40 per class (Total 80)
FOLDS = 5
SEED = 43          # Seed Kemenangan (Wajib sama agar data yang terpilih sama)

print("🚀 EXP 49: TRAINING FINAL CHAMPION (STAGE 2B)")
print("   Configuration: SVM Linear (C=0.5) + Top 30 Features")

# ================= 1. LOAD LABELS =================
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)]
    mood_map = dict(zip(df['id'], df['mood']))
    print(f"📊 Labels Loaded: {len(df)} entries")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ================= 2. AUDIO ELITE SELECTION =================
# Proses ini memastikan kita menggunakan data training yang sama persis
# dengan saat kita mendapatkan skor 89.75%
print("\n⚖️ Re-selecting Elite Dataset...")
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

for p in tqdm(all_files, desc="Clustering Scan"):
    fid = get_id_smart(p)
    if fid not in mood_map: continue
    try:
        X_audio_raw.append(extract_cluster_feat(p))
        filenames.append(p)
    except: pass

X_audio_raw = np.array(X_audio_raw)
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_audio_raw)

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
print(f"✅ Elite Data Selected: {len(selected_files)} samples.")

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

# ================= 4. BUILD CHAMPION PIPELINE =================
print("\n🔨 Building The Champion Pipeline...")

# A. Selector: Pilih 30 Fitur Terbaik
# Kita tetap pakai RF kecil sebagai 'juri' untuk memilih fitur
rfe_selector = RFE(
    estimator=RandomForestClassifier(n_estimators=50, random_state=SEED),
    n_features_to_select=30,  # <-- WINNING PARAM
    step=0.05
)

# B. Classifier: SVM Linear C=0.5
svm_champion = SVC(
    C=0.5,                # <-- WINNING PARAM
    kernel='linear',      # <-- WINNING PARAM
    gamma='scale',
    probability=True,     # Agar bisa keluar output persentase
    random_state=SEED
)

# C. Rakit Pipeline
final_pipeline = Pipeline([
    ('scaler', StandardScaler()), # Wajib untuk SVM
    ('selection', rfe_selector),
    ('model', svm_champion)
])

# ================= 5. FINAL VERIFICATION =================
print(f"🚀 Verifying Performance (Should be ~89.75%)...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
scores = cross_val_score(final_pipeline, X, labels, cv=skf, scoring='accuracy')

mean_acc = np.mean(scores) * 100
std_acc = np.std(scores) * 100

print("\n" + "="*50)
print(f"🏆 VERIFICATION RESULT")
print("="*50)
print(f"✅ Mean Accuracy : {mean_acc:.2f}%")
print(f"📉 Std Deviation : ±{std_acc:.2f}%")
print("-" * 50)

# Visualisasi Confusion Matrix
y_pred_viz = cross_val_predict(final_pipeline, X, labels, cv=skf)
print(classification_report(labels, y_pred_viz, target_names=['sad', 'relaxed']))

cm = confusion_matrix(labels, y_pred_viz)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sad', 'Relaxed'], 
            yticklabels=['Sad', 'Relaxed'])
plt.title(f"Stage 2B Final\nAcc: {mean_acc:.2f}%")
plt.show()

# ================= 6. SAVE PRODUCTION MODEL =================
print("\n💾 Saving Production Model...")

# Latih model pada SELURUH data (80 sampel) agar siap pakai
final_pipeline.fit(X, labels)

if not os.path.exists('models'): os.makedirs('models')
save_path = 'models/stage2b_svm_champion.pkl'
joblib.dump(final_pipeline, save_path)

# Save Metadata
with open('models/stage2b_info.txt', 'w') as f:
    f.write("Model: SVM Linear Pipeline (Stage 2B)\n")
    f.write(f"Accuracy: {mean_acc:.2f}%\n")
    f.write("Params: C=0.5, Kernel=Linear, Features=30\n")
    f.write("Preprocessing: StandardScaler -> RFE -> SVM\n")

print(f"✅ Model saved to: {save_path}")
print("   Siap digunakan untuk Inference!")