"""
=============================================================================
LATENCY BENCHMARK - Hierarchical Mood Prediction System
=============================================================================
Script ini mengukur latensi inferensi model hierarkis untuk setiap stage
dan pipeline secara global.

Pipeline:
  Stage 1 : PyTorch NN (YAMNet features) → High vs Low Energy
  Stage 2A: Stacking RF + MetaLR         → Angry vs Happy  (High Energy)
  Stage 2B: Fine-tuned RoBERTa           → Sad vs Relaxed  (Low Energy)

METRICS:
  - P50 (median), P90, P95 latency (ms) per stage & global
  - Breakdown: feature extraction vs model inference

OUTPUT:
  - Tabel ringkasan latency di console
  - latency_results.csv     → raw per-sample latency
  - latency_summary.csv     → P50 / P90 / P95 per komponen
  - latency_boxplot.png     → boxplot visualisasi
  - latency_breakdown.png   → breakdown feature extraction vs inference

CARA PAKAI:
  1. Jalankan dari folder: machineLearning/
     python comparisson_model/latency/latency_benchmark.py

  2. (Opsional) Ubah N_RUNS dan USE_SYNTHETIC di bagian KONFIGURASI
     - USE_SYNTHETIC=True  : pakai data dummy (tidak perlu data test asli)
     - USE_SYNTHETIC=False : pakai data dari data/data_test/
=============================================================================
"""

import os
import sys
import re
import time
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import librosa

# TF / TF-Hub
import tensorflow as tf
import tensorflow_hub as hub

# HuggingFace
from transformers import RobertaForSequenceClassification, RobertaTokenizer

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# KONFIGURASI — sesuaikan jika perlu
# =============================================================================
# Path (relatif dari folder machineLearning/)
MODEL_DIR         = "models/v3"
STAGE1_NN_PATH    = os.path.join(MODEL_DIR, "stage1_nn.pth")
STAGE1_ENC_PATH   = os.path.join(MODEL_DIR, "stage1_encoder.pkl")
STAGE2A_RF_PATH   = os.path.join(MODEL_DIR, "stage2a_rf.pkl")
STAGE2A_META_PATH = os.path.join(MODEL_DIR, "stage2a_meta.pkl")
STAGE2B_DIR       = os.path.join(MODEL_DIR, "model_roberta_export")

TEST_DATA_DIR = "data/data_test"
LYRICS_PATH   = "data/data_test/lyrics.xlsx"

# Output folder (sama dengan lokasi script ini)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Jumlah run per sample (untuk stabilitas pengukuran)
N_RUNS = 5

# True = gunakan audio sintetis (tidak butuh data test asli)
# False = pakai file audio dari TEST_DATA_DIR
USE_SYNTHETIC = False

# Durasi audio sintetis (detik) — gunakan durasi realistis
SYNTH_AUDIO_DURATION_SEC = 30

# Jumlah sample sintetis yang diukur
N_SYNTHETIC_SAMPLES = 20

TARGET_SR  = 16000
MOODS      = ["angry", "happy", "sad", "relaxed"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM  = 3074   # YAMNet mean+std+max (1024×3) + RMS + ZCR

ROBERTA_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# =============================================================================
# ARSITEKTUR STAGE 1 (harus identik dengan training)
# =============================================================================
class AudioClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )
        self.output = nn.Linear(256, 2)

    def forward(self, x):
        return self.output(self.layer2(self.layer1(x)))


# =============================================================================
# HELPER: timer context manager (ms)
# =============================================================================
class Timer:
    """Pengukur waktu dalam milidetik."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def percentile_str(arr, label=""):
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    p95 = np.percentile(arr, 95)
    return f"{label:30s}  P50={p50:8.2f} ms  P90={p90:8.2f} ms  P95={p95:8.2f} ms"


# =============================================================================
# 1. LOAD MODELS
# =============================================================================
print("=" * 65)
print("🚀  MEMUAT SEMUA MODEL …")
print("=" * 65)

print("⏳  YAMNet …")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

from transformers import pipeline as hf_pipeline
print("⏳  RoBERTa Emotion (Stage 2A text extractor) …")
nlp_emotion = hf_pipeline(
    "text-classification",
    model=ROBERTA_EMOTION_MODEL,
    top_k=None,
    truncation=True,
)

print("⏳  Stage 1 PyTorch NN …")
stage1_encoder = joblib.load(STAGE1_ENC_PATH)
stage1_model   = AudioClassifier(INPUT_DIM).to(DEVICE)
stage1_model.load_state_dict(torch.load(STAGE1_NN_PATH, map_location=DEVICE))
stage1_model.eval()

print("⏳  Stage 2A (RF + MetaLR) …")
stage2a_rf   = joblib.load(STAGE2A_RF_PATH)
stage2a_meta = joblib.load(STAGE2A_META_PATH)

print("⏳  Stage 2B Fine-tuned RoBERTa …")
stage2b_tokenizer = RobertaTokenizer.from_pretrained(STAGE2B_DIR)
stage2b_model     = RobertaForSequenceClassification.from_pretrained(STAGE2B_DIR)
stage2b_model.eval()
stage2b_id2label  = {0: "sad", 1: "relaxed"}

print("✅  Semua model berhasil dimuat.\n")


# =============================================================================
# 2. FUNGSI FEATURE EXTRACTION & INFERENCE
# =============================================================================

def trim_middle(y, sr, pct=0.5):
    if len(y) < sr:
        return y
    start = int(len(y) * (1 - pct) / 2)
    return y[start : start + int(len(y) * pct)]


def feat_stage1(y: np.ndarray):
    y = y.astype(np.float32)
    mx = np.max(np.abs(y))
    y_n = y / mx if mx > 0 else y
    if len(y_n) < TARGET_SR:
        y_n = np.pad(y_n, (0, TARGET_SR - len(y_n)))
    _, emb, _ = yamnet(y_n)
    emb_np = emb.numpy()
    mean_  = emb_np.mean(axis=0)
    std_   = emb_np.std(axis=0)
    max_   = emb_np.max(axis=0)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    return np.concatenate([mean_, std_, max_, [rms, zcr]]).astype(np.float32)


def feat_stage2a_audio(y: np.ndarray):
    y = y.astype(np.float32)
    if len(y) < TARGET_SR:
        y = np.pad(y, (0, TARGET_SR - len(y)))
    _, emb, _ = yamnet(y)
    mean_ = emb.numpy().mean(axis=0)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    return np.concatenate([mean_, [rms, zcr]]).astype(np.float32)


def text_scores_stage2a(lyric: str):
    text   = str(lyric)[:512]
    output = nlp_emotion(text)[0]
    scores = {item["label"]: item["score"] for item in output}
    s_happy = scores.get("joy", 0) + scores.get("surprise", 0)
    s_angry = scores.get("anger", 0) + scores.get("disgust", 0) + scores.get("fear", 0)
    return [s_angry, s_happy]


def clean_lyrics(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    for w in ["lyrics", "embed", "contributors", "translation"]:
        text = text.replace(w, "")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^a-z0-9\s.,'!?]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# 3. PERSIAPAN DATA SAMPLE
# =============================================================================
print("=" * 65)
print("📥  MENYIAPKAN DATA SAMPLE …")
print("=" * 65)

sample_lyrics = (
    "I feel so happy today, everything is going great. "
    "The sun is shining and I can't stop smiling. "
    "Life is beautiful and full of joy."
)

if USE_SYNTHETIC:
    print(f"🎲  Mode: SINTETIS ({N_SYNTHETIC_SAMPLES} sampel, {SYNTH_AUDIO_DURATION_SEC}s/sampel)")
    # Buat audio sintetis: sinyal audio sederhana (campuran sinus)
    n_samples = TARGET_SR * SYNTH_AUDIO_DURATION_SEC
    rng = np.random.default_rng(42)
    synth_audios = [
        rng.uniform(-0.5, 0.5, n_samples).astype(np.float32)
        for _ in range(N_SYNTHETIC_SAMPLES)
    ]
    synth_lyrics = [sample_lyrics] * N_SYNTHETIC_SAMPLES
    data_samples = list(zip(synth_audios, synth_lyrics))
    print(f"✅  {len(data_samples)} sampel sintetis siap.")

else:
    print("📂  Mode: REAL DATA (dari data/data_test/)")
    import glob

    # Load lyrics
    try:
        df_lyr = pd.read_excel(LYRICS_PATH)
        df_lyr.columns = [c.strip().lower() for c in df_lyr.columns]
        df_lyr["id"] = df_lyr["id"].apply(
            lambda x: str(int(float(x))) if str(x).replace(".", "").isdigit() else str(x).strip()
        )
        lyrics_col = "lyrics" if "lyrics" in df_lyr.columns else df_lyr.columns[2]
        lyrics_map = pd.Series(df_lyr[lyrics_col].values, index=df_lyr["id"]).to_dict()
    except Exception as e:
        print(f"⚠️  Gagal load lyrics: {e}")
        lyrics_map = {}

    data_samples = []
    for mood in MOODS:
        folder = os.path.join(TEST_DATA_DIR, mood)
        if not os.path.isdir(folder):
            continue
        files = glob.glob(os.path.join(folder, "*.mp3")) + \
                glob.glob(os.path.join(folder, "*.wav"))
        for fp in files:
            fid   = os.path.basename(fp).split("_")[0].strip()
            lyric = lyrics_map.get(fid, sample_lyrics)
            try:
                y, _ = librosa.load(fp, sr=TARGET_SR)
                if len(y) < TARGET_SR:
                    y = np.pad(y, (0, TARGET_SR - len(y)))
                data_samples.append((y, lyric))
            except Exception as ex:
                print(f"⚠️  Skip {fp}: {ex}")

    if not data_samples:
        print("⚠️  Tidak ada data real. Beralih ke sintetis.")
        n_samples = TARGET_SR * SYNTH_AUDIO_DURATION_SEC
        rng = np.random.default_rng(42)
        data_samples = [
            (rng.uniform(-0.5, 0.5, n_samples).astype(np.float32), sample_lyrics)
            for _ in range(N_SYNTHETIC_SAMPLES)
        ]

    print(f"✅  {len(data_samples)} sampel siap.")

print()

# =============================================================================
# 4. PENGUKURAN LATENCY
# =============================================================================
print("=" * 65)
print(f"⏱️   MENGUKUR LATENCY  (N_RUNS={N_RUNS} per sampel) …")
print("=" * 65)

# Struktur untuk menyimpan hasil
records = []

for idx, (y, lyric) in enumerate(data_samples):
    lyric_clean = clean_lyrics(lyric)
    y_trim = trim_middle(y, TARGET_SR)
    rec = {"sample_id": idx}

    for run in range(N_RUNS):
        # ------------------------------------------------------------------
        # STAGE 1 — Feature Extraction
        # ------------------------------------------------------------------
        with Timer() as t_s1_feat:
            feat_s1 = feat_stage1(y_trim)
        s1_feat_ms = t_s1_feat.elapsed_ms

        # STAGE 1 — Inference
        with Timer() as t_s1_inf:
            with torch.no_grad():
                t_in = torch.tensor(feat_s1).unsqueeze(0).to(DEVICE)
                logits_s1 = stage1_model(t_in)
            probs_s1 = torch.softmax(logits_s1, dim=-1)[0].cpu().numpy()
        s1_inf_ms = t_s1_inf.elapsed_ms

        stage1_pred_idx = int(np.argmax(probs_s1))
        stage1_pred     = stage1_encoder.classes_[stage1_pred_idx]

        # ------------------------------------------------------------------
        # STAGE 2A — (Angry vs Happy)
        # ------------------------------------------------------------------
        with Timer() as t_2a_feat:
            feat_2a = feat_stage2a_audio(y)
        s2a_audio_feat_ms = t_2a_feat.elapsed_ms

        with Timer() as t_2a_txt:
            txt_sc = text_scores_stage2a(lyric_clean)
        s2a_txt_ms = t_2a_txt.elapsed_ms

        with Timer() as t_2a_inf:
            prob_audio = stage2a_rf.predict_proba(feat_2a.reshape(1, -1))[0]
            X_meta     = np.concatenate([prob_audio, txt_sc]).reshape(1, -1)
            _pred_2a   = stage2a_meta.predict(X_meta)[0]
        s2a_inf_ms = t_2a_inf.elapsed_ms

        # ------------------------------------------------------------------
        # STAGE 2B — (Sad vs Relaxed)
        # ------------------------------------------------------------------
        with Timer() as t_2b_feat:
            inputs_2b = stage2b_tokenizer(
                lyric_clean if lyric_clean else "unknown",
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
        s2b_tok_ms = t_2b_feat.elapsed_ms

        with Timer() as t_2b_inf:
            with torch.no_grad():
                logits_2b = stage2b_model(**inputs_2b).logits
            _probs_2b = torch.softmax(logits_2b, dim=-1)[0].numpy()
        s2b_inf_ms = t_2b_inf.elapsed_ms

        # ------------------------------------------------------------------
        # Totals
        # ------------------------------------------------------------------
        s1_total_ms = s1_feat_ms + s1_inf_ms
        s2a_total_ms = s2a_audio_feat_ms + s2a_txt_ms + s2a_inf_ms
        s2b_total_ms = s2b_tok_ms + s2b_inf_ms

        # Total pipeline (Stage 1 selalu jalan, lalu Stage 2A atau 2B)
        if stage1_pred == "high":
            pipeline_ms = s1_total_ms + s2a_total_ms
        else:
            pipeline_ms = s1_total_ms + s2b_total_ms

        records.append({
            "sample_id"          : idx,
            "run"                : run,
            "stage1_pred"        : stage1_pred,
            # Stage 1
            "s1_feat_ms"         : s1_feat_ms,
            "s1_inf_ms"          : s1_inf_ms,
            "s1_total_ms"        : s1_total_ms,
            # Stage 2A
            "s2a_audio_feat_ms"  : s2a_audio_feat_ms,
            "s2a_txt_ms"         : s2a_txt_ms,
            "s2a_inf_ms"         : s2a_inf_ms,
            "s2a_total_ms"       : s2a_total_ms,
            # Stage 2B
            "s2b_tok_ms"         : s2b_tok_ms,
            "s2b_inf_ms"         : s2b_inf_ms,
            "s2b_total_ms"       : s2b_total_ms,
            # Pipeline (Stage 1 + aktual Stage 2)
            "pipeline_ms"        : pipeline_ms,
        })

    if (idx + 1) % 5 == 0 or idx == 0:
        print(f"   ✓ Sampel {idx+1}/{len(data_samples)} selesai diukur")

print("\n✅  Pengukuran selesai.\n")


# =============================================================================
# 5. HITUNG STATISTIK LATENCY
# =============================================================================
df = pd.DataFrame(records)

# Kolom yang diukur & label tampilannya
METRICS = {
    "s1_feat_ms"        : "Stage 1 — Feature Extraction (YAMNet)",
    "s1_inf_ms"         : "Stage 1 — Model Inference (PyTorch NN)",
    "s1_total_ms"       : "Stage 1 — TOTAL",
    "s2a_audio_feat_ms" : "Stage 2A — Audio Feature Extraction",
    "s2a_txt_ms"        : "Stage 2A — Text Scoring (RoBERTa Emotion)",
    "s2a_inf_ms"        : "Stage 2A — Model Inference (RF + MetaLR)",
    "s2a_total_ms"      : "Stage 2A — TOTAL (Angry/Happy)",
    "s2b_tok_ms"        : "Stage 2B — Tokenization",
    "s2b_inf_ms"        : "Stage 2B — Model Inference (RoBERTa)",
    "s2b_total_ms"      : "Stage 2B — TOTAL (Sad/Relaxed)",
    "pipeline_ms"       : "FULL PIPELINE (Stage 1 + aktual Stage 2)",
}

summary_rows = []
print("=" * 65)
print("📊  HASIL LATENCY  (P50 / P90 / P95 dalam ms)")
print("=" * 65)
print()

for col, label in METRICS.items():
    vals = df[col].dropna().values
    p50 = np.percentile(vals, 50)
    p90 = np.percentile(vals, 90)
    p95 = np.percentile(vals, 95)
    mean_v = vals.mean()
    std_v  = vals.std()
    print(percentile_str(vals, label))
    summary_rows.append({
        "component": label,
        "n"        : len(vals),
        "mean_ms"  : round(mean_v, 2),
        "std_ms"   : round(std_v, 2),
        "P50_ms"   : round(p50, 2),
        "P90_ms"   : round(p90, 2),
        "P95_ms"   : round(p95, 2),
        "min_ms"   : round(vals.min(), 2),
        "max_ms"   : round(vals.max(), 2),
    })
    if col in {"s1_total_ms", "s2a_total_ms", "s2b_total_ms"}:
        print()  # separator antar stage

df_summary = pd.DataFrame(summary_rows)

print()
print("=" * 65)
print("📌  Pipeline Summary (P50 / P90 / P95)")
print("=" * 65)
row_pipe = df_summary[df_summary["component"].str.startswith("FULL")].iloc[0]
print(f"   P50 : {row_pipe['P50_ms']:.2f} ms")
print(f"   P90 : {row_pipe['P90_ms']:.2f} ms")
print(f"   P95 : {row_pipe['P95_ms']:.2f} ms")
print(f"   Mean: {row_pipe['mean_ms']:.2f} ms  ±  {row_pipe['std_ms']:.2f} ms")
print()


# =============================================================================
# 6. SIMPAN CSV
# =============================================================================
raw_csv  = os.path.join(OUTPUT_DIR, "latency_results.csv")
summ_csv = os.path.join(OUTPUT_DIR, "latency_summary.csv")

df.to_csv(raw_csv, index=False)
df_summary.to_csv(summ_csv, index=False)

print(f"💾  Raw results  → {raw_csv}")
print(f"💾  Summary      → {summ_csv}")
print()


# =============================================================================
# 7. VISUALISASI
# =============================================================================

# ── 7A. BOXPLOT — latency per stage & pipeline
# Pilih kolom total & pipeline untuk boxplot utama
bp_cols   = ["s1_total_ms", "s2a_total_ms", "s2b_total_ms", "pipeline_ms"]
bp_labels = [
    "Stage 1\n(High/Low)",
    "Stage 2A\n(Angry/Happy)",
    "Stage 2B\n(Sad/Relaxed)",
    "Full\nPipeline",
]
bp_colors = ["#4C9BE8", "#5CB85C", "#9B59B6", "#E67E22"]

fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot(
    [df[c].dropna().values for c in bp_cols],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)

for patch, color in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

# Anotasi P50 / P90 / P95
for i, col in enumerate(bp_cols, start=1):
    vals = df[col].dropna().values
    p50 = np.percentile(vals, 50)
    p90 = np.percentile(vals, 90)
    p95 = np.percentile(vals, 95)
    ax.text(i, p95 * 1.02, f"P95={p95:.0f}", ha="center", va="bottom",
            fontsize=7.5, color="#333")
    ax.text(i, p90 * 1.02, f"P90={p90:.0f}", ha="center", va="bottom",
            fontsize=7.5, color="#555")
    ax.text(i, p50 * 0.98, f"P50={p50:.0f}", ha="center", va="top",
            fontsize=7.5, color="white", fontweight="bold")

ax.set_xticks(range(1, len(bp_cols) + 1))
ax.set_xticklabels(bp_labels, fontsize=11)
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_title(
    "Hierarchical Model — Latency Distribution\n(P50 / P90 / P95 per Stage)",
    fontsize=13, fontweight="bold",
)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
bp_path = os.path.join(OUTPUT_DIR, "latency_boxplot.png")
plt.savefig(bp_path, dpi=150)
plt.close()
print(f"📊  Boxplot       → {bp_path}")


# ── 7B. GROUPED BAR — Feature Extraction vs Inference breakdown
def make_breakdown_chart():
    stages = {
        "Stage 1": {
            "Feature Ext.": df["s1_feat_ms"].dropna(),
            "Inference"   : df["s1_inf_ms"].dropna(),
        },
        "Stage 2A": {
            "Audio Feat." : df["s2a_audio_feat_ms"].dropna(),
            "Text Scoring": df["s2a_txt_ms"].dropna(),
            "Inference"   : df["s2a_inf_ms"].dropna(),
        },
        "Stage 2B": {
            "Tokenization": df["s2b_tok_ms"].dropna(),
            "Inference"   : df["s2b_inf_ms"].dropna(),
        },
    }

    fig2, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    pct_colors = {"P50": "#2196F3", "P90": "#FF9800", "P95": "#F44336"}

    for ax2, (stage_name, comps) in zip(axes, stages.items()):
        comp_names = list(comps.keys())
        x = np.arange(len(comp_names))
        width = 0.25
        for j, (pct_label, pct_val) in enumerate(zip(["P50", "P90", "P95"], [50, 90, 95])):
            bars = [np.percentile(comps[c], pct_val) for c in comp_names]
            rects = ax2.bar(
                x + (j - 1) * width, bars, width,
                label=pct_label, color=pct_colors[pct_label], alpha=0.82,
            )
            for rect in rects:
                h = rect.get_height()
                ax2.text(
                    rect.get_x() + rect.get_width() / 2,
                    h + 0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7,
                )
        ax2.set_title(stage_name, fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(comp_names, fontsize=9, rotation=15, ha="right")
        ax2.set_ylabel("Latency (ms)", fontsize=10)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=9)

    fig2.suptitle(
        "Latency Breakdown per Stage Component  (P50 / P90 / P95)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    bd_path = os.path.join(OUTPUT_DIR, "latency_breakdown.png")
    fig2.savefig(bd_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    return bd_path

bd_path = make_breakdown_chart()
print(f"📊  Breakdown     → {bd_path}")


# ── 7C. LINE CHART — Latency per sampel (trend)
fig3, ax3 = plt.subplots(figsize=(12, 4))
# Rata-rata per sample_id
df_mean = df.groupby("sample_id")[["pipeline_ms", "s1_total_ms", "s2a_total_ms", "s2b_total_ms"]].mean()

ax3.plot(df_mean.index, df_mean["pipeline_ms"], label="Full Pipeline", linewidth=2, marker="o", ms=4)
ax3.plot(df_mean.index, df_mean["s1_total_ms"], label="Stage 1", linewidth=1.5, linestyle="--")
ax3.plot(df_mean.index, df_mean["s2a_total_ms"], label="Stage 2A", linewidth=1.5, linestyle=":")
ax3.plot(df_mean.index, df_mean["s2b_total_ms"], label="Stage 2B", linewidth=1.5, linestyle="-.")

# Hline P95
p95_pipe = np.percentile(df["pipeline_ms"].dropna(), 95)
ax3.axhline(p95_pipe, color="red", linestyle="--", linewidth=1.2, label=f"P95 Pipeline = {p95_pipe:.0f} ms")

ax3.set_xlabel("Sample Index", fontsize=11)
ax3.set_ylabel("Latency (ms)", fontsize=11)
ax3.set_title("Latency per Sample (Mean over Runs)", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)
ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)
plt.tight_layout()
lc_path = os.path.join(OUTPUT_DIR, "latency_per_sample.png")
fig3.savefig(lc_path, dpi=150)
plt.close(fig3)
print(f"📊  Per-sample    → {lc_path}")


# =============================================================================
# 8. RINGKASAN AKHIR
# =============================================================================
print()
print("=" * 65)
print("📋  RINGKASAN LATENCY FINAL")
print("=" * 65)
print(f"   {'Komponen':<40} {'P50':>8} {'P90':>8} {'P95':>8}")
print("   " + "-" * 60)
for _, row in df_summary.iterrows():
    print(f"   {row['component']:<40} {row['P50_ms']:>7.1f}  {row['P90_ms']:>7.1f}  {row['P95_ms']:>7.1f}")
print()
print("   ✅  Benchmark selesai!")
print(f"   📁  Output: {OUTPUT_DIR}")
print("=" * 65)
