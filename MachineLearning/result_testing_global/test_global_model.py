"""
=============================================================================
TEST GLOBAL MODEL - Hierarchical Mood Prediction System
=============================================================================
Script ini digunakan untuk mengevaluasi performa model hierarkis global
yang terdiri dari 3 model:

  Stage 1: PyTorch NN       → High vs Low Energy
  Stage 2A: Stacking RF+LR  → Angry vs Happy  (jika High Energy)
  Stage 2B: Fine-tuned RoBERTa → Sad vs Relaxed (jika Low Energy)

DATA TEST:
  Audio : data/data_test/{mood}/id_filename.mp3
  Lyrics: data/data_test/lyrics.xlsx  (kolom: id, name, lyric)

MODELS:
  models/v3/stage1_nn.pth
  models/v3/stage1_encoder.pkl
  models/v3/stage2a_rf.pkl
  models/v3/stage2a_meta.pkl
  models/v3/model_roberta_export/   (fine-tuned RoBERTa untuk Sad/Relaxed)

OUTPUT:
  - Validation score (accuracy, F1, classification report) per stage & global
  - Confusion matrix (disimpan sebagai PNG di folder results/)
=============================================================================
"""

import os
import re
import glob
import logging

# Mute TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import librosa
import joblib
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_hub as hub

from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# KONFIGURASI
# =============================================================================
# Path ke folder data test
TEST_DATA_DIR  = '../data/data_test'
LYRICS_PATH    = '../data/data_test/lyrics.xlsx'

# Path ke model v3
MODEL_DIR          = '../models/v3'
STAGE1_NN_PATH     = os.path.join(MODEL_DIR, 'stage1_nn.pth')
STAGE1_ENC_PATH    = os.path.join(MODEL_DIR, 'stage1_encoder.pkl')
STAGE2A_RF_PATH    = os.path.join(MODEL_DIR, 'stage2a_rf.pkl')
STAGE2A_META_PATH  = os.path.join(MODEL_DIR, 'stage2a_meta.pkl')
STAGE2B_MODEL_DIR  = os.path.join(MODEL_DIR, 'model_roberta_export')

# Hasil output
RESULTS_DIR = '../result_testing_global'
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SR = 16000
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mood → label final
MOODS = ['angry', 'happy', 'sad', 'relaxed']

# Model RoBERTa untuk ekstraksi fitur teks Stage 2A
ROBERTA_EMOTION_MODEL = 'j-hartmann/emotion-english-distilroberta-base'

# =============================================================================
# DEFINISI ARSITEKTUR MODEL STAGE 1 (harus sama persis dengan training)
# =============================================================================
class AudioClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AudioClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(256, 2)  # 2 kelas: high vs low

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)


# =============================================================================
# 1. LOAD MODELS
# =============================================================================
print('=' * 60)
print('🚀 MEMUAT SEMUA MODEL...')
print('=' * 60)

# 1a. YAMNet (shared feature extractor)
print('⏳ Loading YAMNet...')
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# 1b. RoBERTa Emotion (shared text extractor untuk Stage 2A)
print('⏳ Loading RoBERTa Emotion Pipeline (Stage 2A text)...')
nlp_emotion = pipeline(
    'text-classification',
    model=ROBERTA_EMOTION_MODEL,
    top_k=None,
    truncation=True
)

# 1c. Stage 1: PyTorch NN
print('⏳ Loading Stage 1 PyTorch NN...')
stage1_encoder = joblib.load(STAGE1_ENC_PATH)
# Input dim = 1024*3 + 2 = 3074
INPUT_DIM = 3074
stage1_model = AudioClassifier(INPUT_DIM).to(DEVICE)
stage1_model.load_state_dict(torch.load(STAGE1_NN_PATH, map_location=DEVICE))
stage1_model.eval()

# 1d. Stage 2A: Stacking RF (audio base) + Logistic Regression (meta)
print('⏳ Loading Stage 2A (RF + MetaLR)...')
stage2a_rf   = joblib.load(STAGE2A_RF_PATH)
stage2a_meta = joblib.load(STAGE2A_META_PATH)

# 1e. Stage 2B: Fine-tuned RoBERTa (Sad vs Relaxed)
print('⏳ Loading Stage 2B Fine-tuned RoBERTa (Sad/Relaxed)...')
stage2b_tokenizer = RobertaTokenizer.from_pretrained(STAGE2B_MODEL_DIR)
stage2b_model     = RobertaForSequenceClassification.from_pretrained(STAGE2B_MODEL_DIR)
stage2b_model.eval()

# Force hardcoded label mapping
# Berdasarkan kode training (lyrics_stage2b_colab.ipynb):
#   MIR Dataset: Q3 (Sad) = 0, Q4 (Relaxed) = 1
#   User Excel : sad = 0, relaxed = 1
# → Label 0 = 'sad', Label 1 = 'relaxed'
stage2b_id2label = {0: 'sad', 1: 'relaxed'}
print(f'   Stage 2B labels (hardcoded): {stage2b_id2label}')

print('✅ SEMUA MODEL BERHASIL DIMUAT\n')


# =============================================================================
# 2. FUNGSI FEATURE EXTRACTION
# =============================================================================

def trim_middle(y, sr, percentage=0.5):
    """Potong bagian tengah audio (sama seperti saat training Stage 1)."""
    if len(y) < sr:
        return y
    start = int(len(y) * (1 - percentage) / 2)
    end   = start + int(len(y) * percentage)
    return y[start:end]


def extract_stage1_features(y):
    """
    Fitur Stage 1: YAMNet mean/std/max embeddings + RMS + ZCR.
    Input y sudah di-trim_middle.
    Total = 1024*3 + 2 = 3074 fitur.
    """
    try:
        y = y.astype(np.float32)
        if np.max(np.abs(y)) > 0:
            y_norm = y / np.max(np.abs(y))
        else:
            y_norm = y
        if len(y_norm) < TARGET_SR:
            y_norm = np.pad(y_norm, (0, TARGET_SR - len(y_norm)))

        _, embeddings, _ = yamnet(y_norm)
        emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        emb_std  = tf.math.reduce_std(embeddings, axis=0).numpy()
        emb_max  = tf.reduce_max(embeddings, axis=0).numpy()

        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

        return np.concatenate([emb_mean, emb_std, emb_max, [rms, zcr]]).astype(np.float32)
    except Exception:
        return None


def extract_stage2a_audio_features(y):
    """
    Fitur audio Stage 2A: YAMNet mean + RMS + ZCR.
    Total = 1024 + 2 = 1026 fitur.
    """
    try:
        y = y.astype(np.float32)
        if len(y) < TARGET_SR:
            y = np.pad(y, (0, TARGET_SR - len(y)))
        _, emb, _ = yamnet(y)
        yamnet_vec = tf.reduce_mean(emb, axis=0).numpy()
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        return np.concatenate([yamnet_vec, [rms, zcr]]).astype(np.float32)
    except Exception:
        return None


def get_text_scores_stage2a(lyric):
    """
    Skor teks untuk Stage 2A: [s_angry, s_happy].
    Menggunakan RoBERTa Emotion (j-hartmann/emotion-english-distilroberta-base).
    """
    try:
        text = str(lyric)[:512]
        output = nlp_emotion(text)[0]
        scores = {item['label']: item['score'] for item in output}
        s_happy = scores.get('joy', 0) + scores.get('surprise', 0)
        s_angry = scores.get('anger', 0) + scores.get('disgust', 0) + scores.get('fear', 0)
        return [s_angry, s_happy]
    except Exception:
        return [0.5, 0.5]


def clean_lyrics(text):
    """Bersihkan teks lirik (sama seperti saat training Stage 2A/2B)."""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    for w in ['lyrics', 'embed', 'contributors', 'translation']:
        text = text.replace(w, '')
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"[^a-z0-9\s.,'!?]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_stage2b(lyric):
    """
    Prediksi Stage 2B: Sad vs Relaxed menggunakan fine-tuned RoBERTa.
    Returns: ('sad' atau 'relaxed', confidence[0..1])
    """
    text = clean_lyrics(lyric)
    if not text:
        text = 'unknown'
    inputs = stage2b_tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        logits = stage2b_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    pred_id  = int(np.argmax(probs))
    pred_lbl = stage2b_id2label[pred_id].lower()
    return pred_lbl, float(probs[pred_id])


# =============================================================================
# 3. LOAD DATA TEST
# =============================================================================
print('=' * 60)
print('📥 LOADING DATA TEST...')
print('=' * 60)

# Load lyrics
try:
    df_lyrics = pd.read_excel(LYRICS_PATH)
    # Normalisasi nama kolom: strip spasi + lowercase
    df_lyrics.columns = [c.strip().lower() for c in df_lyrics.columns]
    print(f'   Kolom ditemukan di lyrics.xlsx: {list(df_lyrics.columns)}')
    df_lyrics['id'] = df_lyrics['id'].apply(lambda x: str(int(float(x))) if str(x).replace('.','').isdigit() else str(x).strip())
    lyrics_map = pd.Series(df_lyrics['lyrics'].values, index=df_lyrics['id']).to_dict()
    print(f'✅ Lyrics loaded: {len(lyrics_map)} entri')
except Exception as e:
    print(f'⚠️  Gagal load lyrics: {e}')
    lyrics_map = {}

# Kumpulkan semua file audio beserta label
records = []
for mood in MOODS:
    folder = os.path.join(TEST_DATA_DIR, mood)
    if not os.path.exists(folder):
        print(f'⚠️  Folder tidak ditemukan: {folder}')
        continue
    files = glob.glob(os.path.join(folder, '*.mp3')) + \
            glob.glob(os.path.join(folder, '*.wav'))
    for fp in files:
        # Ambil ID dari nama file (format: id_filename.ext)
        basename = os.path.basename(fp)
        fid = basename.split('_')[0].strip()
        lyric = lyrics_map.get(fid, '')
        records.append({
            'path': fp,
            'id': fid,
            'true_mood': mood,
            'lyric': lyric
        })

print(f'✅ Total file audio ditemukan: {len(records)}')
for mood in MOODS:
    n = sum(1 for r in records if r['true_mood'] == mood)
    print(f'   {mood}: {n} file')
print()


# =============================================================================
# 4. PREDIKSI TIAP FILE (HIERARCHICAL PIPELINE)
# =============================================================================
print('=' * 60)
print('🔮 PREDIKSI HIERARCHICAL MODEL...')
print('=' * 60)

results = []

for rec in tqdm(records, desc='Predicting'):
    path      = rec['path']
    true_mood = rec['true_mood']
    lyric     = rec['lyric']
    lyric_clean = clean_lyrics(lyric)

    # --- Load Audio ---
    try:
        y, sr = librosa.load(path, sr=TARGET_SR)
        if len(y) < TARGET_SR:
            y = np.pad(y, (0, TARGET_SR - len(y)))
    except Exception as e:
        results.append({**rec, 'stage1_pred': None, 'final_pred': None, 'error': str(e)})
        continue

    # -------------------------------------------------------------------------
    # STAGE 1: High vs Low Energy
    # -------------------------------------------------------------------------
    y_trimmed = trim_middle(y, sr)
    feat_s1   = extract_stage1_features(y_trimmed)
    if feat_s1 is None:
        results.append({**rec, 'stage1_pred': None, 'final_pred': None, 'error': 'stage1 feat fail'})
        continue

    with torch.no_grad():
        t = torch.tensor(feat_s1, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits_s1 = stage1_model(t)

    probs_s1 = torch.softmax(logits_s1, dim=-1)[0].cpu().numpy()
    # Stage 1 encoder: classes_[0]=high, classes_[1]=low  (urutan sesuai LabelEncoder)
    stage1_pred_idx = int(np.argmax(probs_s1))
    stage1_pred_lbl = stage1_encoder.classes_[stage1_pred_idx]  # 'high' atau 'low'

    # Tentukan true stage1 label dari mood
    true_stage1 = 'high' if true_mood in ['angry', 'happy'] else 'low'

    # -------------------------------------------------------------------------
    # STAGE 2: Angry/Happy  ATAU  Sad/Relaxed
    # -------------------------------------------------------------------------
    if stage1_pred_lbl == 'high':
        # --- Stage 2A: Angry vs Happy ---
        feat_s2a_audio = extract_stage2a_audio_features(y)
        if feat_s2a_audio is None:
            final_pred = None
            results.append({**rec,
                            'stage1_pred': stage1_pred_lbl,
                            'true_stage1': true_stage1,
                            'final_pred': None,
                            'error': 'stage2a feat fail'})
            continue

        # Base: RF predict_proba
        prob_audio = stage2a_rf.predict_proba(feat_s2a_audio.reshape(1, -1))[0]  # [P_angry, P_happy]

        # Text scores
        txt_scores = get_text_scores_stage2a(lyric_clean)  # [s_angry, s_happy]

        # Meta input = [P_angry_audio, P_happy_audio, s_angry_txt, s_happy_txt]
        X_meta = np.concatenate([prob_audio, txt_scores]).reshape(1, -1)
        final_pred = stage2a_meta.predict(X_meta)[0]
        # Map 0/1 → 'angry'/'happy' (sesuai training Stage 2A)
        final_pred = 'angry' if final_pred == 0 else 'happy'

    else:
        # --- Stage 2B: Sad vs Relaxed ---
        final_pred, _ = predict_stage2b(lyric_clean)

    results.append({
        'path': path,
        'id': rec['id'],
        'true_mood': true_mood,
        'true_stage1': true_stage1,
        'stage1_pred': stage1_pred_lbl,
        'final_pred': final_pred,
        'error': None
    })

print(f'\n✅ Selesai. Berhasil diproses: {sum(1 for r in results if r.get("error") is None)} / {len(results)}')
errors = [r for r in results if r.get('error')]
if errors:
    print(f'⚠️  Gagal diproses: {len(errors)} file')
    for e in errors[:5]:
        print(f'   {e["path"]}: {e["error"]}')


# =============================================================================
# 5. EVALUASI & CONFUSION MATRIX
# =============================================================================
print('\n' + '=' * 60)
print('📊 EVALUASI HASIL')
print('=' * 60)

valid = [r for r in results if r.get('error') is None and r.get('final_pred') is not None]

# ---- 5A. STAGE 1 EVALUATION ----
y_true_s1 = [r['true_stage1'] for r in valid]
y_pred_s1 = [r['stage1_pred'] for r in valid]

acc_s1 = accuracy_score(y_true_s1, y_pred_s1)
f1_s1  = f1_score(y_true_s1, y_pred_s1, average='weighted')

print('\n📌 Stage 1 (High vs Low Energy)')
print(f'   Accuracy : {acc_s1*100:.2f}%')
print(f'   F1 Score : {f1_s1:.4f}')
print(classification_report(y_true_s1, y_pred_s1, target_names=['high', 'low']))

cm_s1 = confusion_matrix(y_true_s1, y_pred_s1, labels=['high', 'low'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_s1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['high', 'low'], yticklabels=['high', 'low'])
plt.title(f'Stage 1 - High vs Low\nAcc: {acc_s1*100:.1f}%  F1: {f1_s1:.3f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
cm_s1_path = os.path.join(RESULTS_DIR, 'cm_test_stage1.png')
plt.savefig(cm_s1_path)
plt.show()
print(f'   💾 CM Stage 1 disimpan: {cm_s1_path}')

# ---- 5B. STAGE 2A EVALUATION (Angry vs Happy) ----
valid_2a = [r for r in valid if r['true_mood'] in ['angry', 'happy']]
if valid_2a:
    y_true_2a = [r['true_mood']  for r in valid_2a]
    y_pred_2a = [r['final_pred'] for r in valid_2a]

    acc_2a = accuracy_score(y_true_2a, y_pred_2a)
    f1_2a  = f1_score(y_true_2a, y_pred_2a, average='weighted')

    print('\n📌 Stage 2A (Angry vs Happy)')
    print(f'   Accuracy : {acc_2a*100:.2f}%')
    print(f'   F1 Score : {f1_2a:.4f}')
    print(classification_report(y_true_2a, y_pred_2a, labels=['angry', 'happy'], target_names=['angry', 'happy'], zero_division=0))

    cm_2a = confusion_matrix(y_true_2a, y_pred_2a, labels=['angry', 'happy'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_2a, annot=True, fmt='d', cmap='Greens',
                xticklabels=['angry', 'happy'], yticklabels=['angry', 'happy'])
    plt.title(f'Stage 2A - Angry vs Happy\nAcc: {acc_2a*100:.1f}%  F1: {f1_2a:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_2a_path = os.path.join(RESULTS_DIR, 'cm_test_stage2a.png')
    plt.savefig(cm_2a_path)
    plt.show()
    print(f'   💾 CM Stage 2A disimpan: {cm_2a_path}')
else:
    print('\n⚠️  Tidak ada sampel Angry/Happy dalam data test.')

# ---- 5C. STAGE 2B EVALUATION (Sad vs Relaxed) ----
valid_2b = [r for r in valid if r['true_mood'] in ['sad', 'relaxed']]
if valid_2b:
    y_true_2b = [r['true_mood']  for r in valid_2b]
    y_pred_2b = [r['final_pred'] for r in valid_2b]

    acc_2b = accuracy_score(y_true_2b, y_pred_2b)
    f1_2b  = f1_score(y_true_2b, y_pred_2b, average='weighted')

    print('\n📌 Stage 2B (Sad vs Relaxed)')
    print(f'   Accuracy : {acc_2b*100:.2f}%')
    print(f'   F1 Score : {f1_2b:.4f}')
    print(classification_report(y_true_2b, y_pred_2b, labels=['sad', 'relaxed'], target_names=['sad', 'relaxed'], zero_division=0))

    cm_2b = confusion_matrix(y_true_2b, y_pred_2b, labels=['sad', 'relaxed'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_2b, annot=True, fmt='d', cmap='Purples',
                xticklabels=['sad', 'relaxed'], yticklabels=['sad', 'relaxed'])
    plt.title(f'Stage 2B - Sad vs Relaxed\nAcc: {acc_2b*100:.1f}%  F1: {f1_2b:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_2b_path = os.path.join(RESULTS_DIR, 'cm_test_stage2b.png')
    plt.savefig(cm_2b_path)
    plt.show()
    print(f'   💾 CM Stage 2B disimpan: {cm_2b_path}')
else:
    print('\n⚠️  Tidak ada sampel Sad/Relaxed dalam data test.')

# ---- 5D. GLOBAL EVALUATION (4-class: angry, happy, sad, relaxed) ----
y_true_global = [r['true_mood']  for r in valid]
y_pred_global = [r['final_pred'] for r in valid]

acc_global = accuracy_score(y_true_global, y_pred_global)
f1_global  = f1_score(y_true_global, y_pred_global, average='weighted')

print('\n' + '=' * 60)
print('🌐 GLOBAL EVALUATION (4-Class: angry, happy, sad, relaxed)')
print('=' * 60)
print(f'   Accuracy : {acc_global*100:.2f}%')
print(f'   F1 Score : {f1_global:.4f}')
print(classification_report(y_true_global, y_pred_global,
                             labels=MOODS, target_names=MOODS, zero_division=0))

cm_global = confusion_matrix(y_true_global, y_pred_global, labels=MOODS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues',
            xticklabels=MOODS, yticklabels=MOODS)
plt.title(
    f'Global Confusion Matrix - 4 Mood Classes\n'
    f'Acc: {acc_global*100:.1f}%  F1: {f1_global:.3f}'
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
cm_global_path = os.path.join(RESULTS_DIR, 'cm_test_global.png')
plt.savefig(cm_global_path)
plt.show()
print(f'\n💾 CM Global disimpan: {cm_global_path}')

# ---- 5E. SIMPAN DETAIL HASIL ----
df_results = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, 'test_detail_results.csv')
df_results.to_csv(csv_path, index=False)
print(f'💾 Detail hasil disimpan: {csv_path}')

print('\n' + '=' * 60)
print('✅ EVALUASI SELESAI!')
print(f'   Stage 1    Acc: {acc_s1*100:.2f}%   F1: {f1_s1:.4f}')
if valid_2a:
    print(f'   Stage 2A   Acc: {acc_2a*100:.2f}%   F1: {f1_2a:.4f}')
if valid_2b:
    print(f'   Stage 2B   Acc: {acc_2b*100:.2f}%   F1: {f1_2b:.4f}')
print(f'   Global     Acc: {acc_global*100:.2f}%   F1: {f1_global:.4f}')
print('=' * 60)
