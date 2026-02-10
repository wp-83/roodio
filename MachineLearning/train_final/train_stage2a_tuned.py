import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import joblib 
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Mute Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- KONFIGURASI ---
SOURCE_DIRS = ['data/raw', 'data/raw2'] 
LYRICS_PATH = 'data/lyrics/lyrics.xlsx' 
TARGET_MOODS = ['angry', 'happy'] 
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
FOLDS = 5
SEED = 43

print(f"🚀 MEMULAI EXP 27 VARIATION: STACKING ENSEMBLE TUNED (ANGRY vs HAPPY)...")

# --- 1. SETUP & CLEANING DATA (SAMA SEPERTI SEBELUMNYA) ---
try:
    df = pd.read_excel(LYRICS_PATH)
    df['id'] = df['id'].astype(str).str.strip()
    df['mood'] = df['mood'].str.lower().str.strip()
    df = df[df['mood'].isin(TARGET_MOODS)].copy()
    lyrics_map = pd.Series(df.lyrics.values, index=df.id).to_dict()
    mood_map = pd.Series(df.mood.values, index=df.id).to_dict()
    print(f"📊 Data Excel Loaded: {len(df)} baris valid.")
except Exception as e:
    print(f"❌ Error loading Excel: {e}")
    exit()

def clean_lyrics_text(text):
    if pd.isna(text) or text == '': return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text) 
    text = re.sub(r'\(.*?\)', ' ', text) 
    garbage = ['lyrics', 'embed', 'contributors', 'translation']
    for w in garbage: text = text.replace(w, '')
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for pid in lyrics_map:
    lyrics_map[pid] = clean_lyrics_text(lyrics_map[pid])

print("✅ Lirik berhasil dibersihkan.")
print("⏳ Loading Models (YAMNet & RoBERTa)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
nlp_classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

# --- 2. FEATURE EXTRACTION (SAMA SEPERTI SEBELUMNYA) ---
X_audio_features = []
X_text_scores = [] 
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
        lyrics_chunk = str(lyrics)[:512]
        output = nlp_classifier(lyrics_chunk)[0]
        scores = {item['label']: item['score'] for item in output}
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
        return [s_angry, s_happy] 
    except: return [0.5, 0.5]

def get_id_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1: return parts[0].strip()
    return None

all_audio_files = []
for d in SOURCE_DIRS:
    all_audio_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    all_audio_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))

for file_path in tqdm(all_audio_files, desc="Processing Audio"):
    fid = get_id_from_filename(file_path)
    if fid not in lyrics_map: continue 
    
    mood = mood_map[fid]
    lyric = lyrics_map[fid]
    
    if mood == 'angry': label = 0
    elif mood == 'happy': label = 1
    else: continue 
    
    aud = extract_audio(file_path)
    txt = get_text_scores(lyric)
    
    if aud is not None:
        X_audio_features.append(aud)
        X_text_scores.append(txt)
        y_labels.append(label)

X_audio_features = np.array(X_audio_features)
X_text_scores = np.array(X_text_scores) 
y_labels = np.array(y_labels)

print(f"✅ Data Siap: {len(y_labels)} sampel.")

# --- 3. STACKING TRAINING (TUNED VERSION) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
acc_scores = []
y_true_all = []
y_pred_all = []
meta_weights = []

print(f"\n🚀 START STACKING TRAINING ({FOLDS}-Fold) WITH TUNING...")

clf_audio_final = None
meta_clf_final = None

for fold, (train_idx, test_idx) in enumerate(skf.split(X_audio_features, y_labels)):
    
    X_aud_tr, X_aud_ts = X_audio_features[train_idx], X_audio_features[test_idx]
    y_tr, y_ts = y_labels[train_idx], y_labels[test_idx]
    X_txt_tr = X_text_scores[train_idx]
    X_txt_ts = X_text_scores[test_idx]
    
    # === VARIASI 1: BASE MODEL YANG LEBIH KUAT ===
    # Random Forest yang lebih robust (Anti-Overfit)
    # clf_audio = RandomForestClassifier(
    #     n_estimators=300,       # Lebih banyak pohon
    #     max_depth=5,            # Pohon pendek (Mencegah overfit)
    #     min_samples_split=4,    # Minimal sampel untuk membelah node
    #     min_samples_leaf=2,     # Minimal sampel di daun
    #     max_features='sqrt',
    #     random_state=SEED,
    #     n_jobs=-1               # Gunakan semua core CPU
    # )
    
    # Opsi Alternatif: Gradient Boosting (Biasanya lebih akurat dari RF)
    clf_audio = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED)
    
    clf_audio.fit(X_aud_tr, y_tr)
    
    # Generate Probabilities untuk Meta-Learner
    # cross_val_predict penting agar data train meta-learner tidak bocor
    prob_audio_train_cv = cross_val_predict(clf_audio, X_aud_tr, y_tr, cv=3, method='predict_proba')
    prob_audio_test = clf_audio.predict_proba(X_aud_ts)
    
    # === VARIASI 2: INPUT META-LEARNER ===
    # Input ke Meta Learner: [Prob Angry (Audio), Prob Happy (Audio), Score Angry (Text), Score Happy (Text)]
    X_meta_train = np.concatenate([prob_audio_train_cv, X_txt_tr], axis=1)
    X_meta_test = np.concatenate([prob_audio_test, X_txt_ts], axis=1)
    
    # === VARIASI 3: META LEARNER YANG LEBIH BIJAK ===
    # Logistic Regression dengan Regularisasi Kuat (C Kecil)
    # Ini memaksa model untuk tidak terlalu percaya pada satu fitur saja
    # meta_clf = LogisticRegression(
    #     C=0.5,                 # Regularisasi lebih kuat dari default (1.0)
    #     solver='liblinear',    # Bagus untuk dataset kecil
    #     class_weight='balanced', # Menangani jika jumlah Angry/Happy tidak seimbang
    #     random_state=SEED
    # )
    
    # Opsi Alternatif: SVM (Jika Logistic Regression kurang bagus)
    meta_clf = SVC(kernel='linear', probability=True, random_state=SEED)
    
    meta_clf.fit(X_meta_train, y_tr)
    
    # Simpan bobot untuk analisis
    # Jika pakai LogisticRegression, kita bisa lihat coef_
    if hasattr(meta_clf, 'coef_'):
        meta_weights.append(meta_clf.coef_[0]) 
        
    y_pred_fold = meta_clf.predict(X_meta_test)
    
    acc = accuracy_score(y_ts, y_pred_fold)
    acc_scores.append(acc)
    print(f"   Fold {fold+1}: {acc*100:.2f}%")
    
    y_true_all.extend(y_ts)
    y_pred_all.extend(y_pred_fold)
    
    clf_audio_final = clf_audio
    meta_clf_final = meta_clf

# --- 4. REPORT & ANALYSIS ---
mean_acc = np.mean(acc_scores) * 100
std_acc = np.std(acc_scores) * 100

print("\n" + "="*50)
print("📊 HASIL AKHIR TUNING STAGE 2A")
print("="*50)

print(f"🏆 Rata-rata Akurasi : {mean_acc:.2f}%")
print(f"📉 Standar Deviasi   : ±{std_acc:.2f}%")

if std_acc < 5.0: print("   ✅ STATUS: SANGAT STABIL")
elif std_acc < 10.0: print("   ⚠️ STATUS: CUKUP STABIL")
else: print("   ❌ STATUS: TIDAK STABIL")

print("-" * 50)
print(classification_report(y_true_all, y_pred_all, target_names=['angry', 'happy']))

# --- 5. ANALISIS BOBOT (FEATURE IMPORTANCE) ---
if len(meta_weights) > 0:
    print("\n⚖️ ANALISIS KONTRIBUSI FITUR (META-LEARNER)")
    avg_weights = np.mean(meta_weights, axis=0)
    
    # Bobot Logistic Regression: [Audio_Angry, Audio_Happy, Text_Angry, Text_Happy]
    # Kita absolutkan agar tahu besarnya pengaruh tanpa peduli arah (+/-)
    w_aud_ang, w_aud_hap, w_txt_ang, w_txt_hap = abs(avg_weights)
    
    audio_score = w_aud_ang + w_aud_hap
    text_score = w_txt_ang + w_txt_hap
    total = audio_score + text_score
    
    print(f"🔊 Audio Importance : {audio_score:.4f} ({(audio_score/total)*100:.1f}%)")
    print(f"📝 Text Importance  : {text_score:.4f} ({(text_score/total)*100:.1f}%)")
    
    # Visualisasi Sederhana
    features = ['Aud_Angry', 'Aud_Happy', 'Txt_Angry', 'Txt_Happy']
    plt.figure(figsize=(8, 4))
    plt.bar(features, abs(avg_weights), color=['blue', 'blue', 'green', 'green'])
    plt.title('Seberapa Penting Audio vs Teks bagi Model?')
    plt.ylabel('Bobot Absolut')
    plt.savefig('feature_importance_stage2a.png')
    plt.close()

# --- 6. SAVING MODEL ---
print("\n💾 Saving Final Models...")
if clf_audio_final and meta_clf_final:
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(clf_audio_final, 'models/stage2a_rf.pkl')
    joblib.dump(meta_clf_final, 'models/stage2a_meta.pkl')
    print("✅ Model Stage 2A (RF + Meta) berhasil disimpan!")
else:
    print("❌ Gagal menyimpan model.")