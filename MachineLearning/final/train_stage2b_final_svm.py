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

# ================= CONFIG =================
SOURCE_DIRS = ['data/raw', 'data/raw2']
LYRICS_PATH = 'data/lyrics/lyrics.xlsx'
TARGET_MOODS = ['sad', 'relaxed']
TARGET_COUNT = 40  # Sesuai experiment sebelumnya (Total 80 Data)
FOLDS = 6
SEED = 43          # Seed kemenangan

print("🚀 EXP 47: TRAINING FINAL CHAMPION (SVM LINEAR)")
print(f"   Target Accuracy: ~88.85%")

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

# ================= 4. BUILD FINAL PIPELINE =================
print("\n🔨 Building Champion Pipeline...")

# A. Feature Selector (RFE)
# Kita tetap pakai RF kecil untuk memilih fitur, karena RF jago melihat importance
rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=50, random_state=SEED),
    n_features_to_select=30, 
    step=0.05
)

# B. Classifier (THE CHAMPION)
# Params: {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'linear'}
svm_champion = SVC(
    C=0.1, 
    kernel='linear', 
    gamma='scale', 
    probability=True, # Wajib True agar bisa keluarin confidence score (%)
    random_state=SEED
)

# C. Pipeline Assembly
# Urutan: Scaler -> RFE -> SVM
final_pipeline = Pipeline([
    ('scaler', StandardScaler()), # SVM Wajib scaling
    ('selection', rfe),           # Pilih 30 fitur terbaik
    ('model', svm_champion)       # Klasifikasi
])

# ================= 5. VERIFICATION (CROSS VALIDATION) =================
print(f"🚀 Verifying Performance (5-Fold CV)...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
scores = cross_val_score(final_pipeline, X, labels, cv=skf, scoring='accuracy')

mean_acc = np.mean(scores) * 100
std_acc = np.std(scores) * 100

print("\n" + "="*50)
print(f"🏆 FINAL VERIFICATION RESULT")
print("="*50)
print(f"✅ Mean Accuracy : {mean_acc:.2f}%")
print(f"📉 Std Deviation : ±{std_acc:.2f}%")
print(f"📊 Fold Scores   : {scores}")

# Confusion Matrix Validation
y_pred_cv = cross_val_predict(final_pipeline, X, labels, cv=skf)
print("\n" + classification_report(labels, y_pred_cv, target_names=['sad', 'relaxed']))

cm = confusion_matrix(labels, y_pred_cv)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sad', 'Relaxed'], 
            yticklabels=['Sad', 'Relaxed'])
plt.title(f"Final SVM Linear\nAcc: {mean_acc:.2f}%")
plt.show()

# ================= 6. SAVE MODEL =================
print("\n💾 Training on FULL DATA & Saving...")

# Fit pipeline ke seluruh data (80 sampel)
final_pipeline.fit(X, labels)

# Save
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(final_pipeline, 'models/stage2b_final_svm.pkl')

# Save Metadata
with open('models/stage2b_info.txt', 'w') as f:
    f.write("Model: SVM Linear (Pipeline)\n")
    f.write(f"Accuracy: {mean_acc:.2f}%\n")
    f.write(f"Components: StandardScaler -> RFE(30) -> SVC(C=0.1, Linear)\n")
    f.write(f"Elite Data: Top 40 per class\n")

print("✅ DONE! Model saved to: models/stage2b_final_svm.pkl")
print("   Siap digunakan untuk Inference!")