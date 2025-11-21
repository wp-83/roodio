# src/config.py
import os

# --- PATHS ---
# Kita gunakan os.path.abspath agar pathnya selalu benar dimanapun dijalankan
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw", "audio")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "metadata", "audio")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "audio")

# --- AUDIO SETTINGS ---
SAMPLE_RATE = 22050   # Downsample ke 22kHz untuk efisiensi
DURATION = 45         # Durasi asli file
SKIP_SECONDS = 15     # Skip 15 detik pertama (karena tidak ada label)
SEGMENT_LENGTH = 3    # Panjang potongan per data latih (detik)

# --- SPECTROGRAM SETTINGS ---
# Setting ini menghasilkan gambar output ukuran 128x130 (cocok untuk CNN)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 8000

# --- LABELING SETTINGS ---
# Label di CSV tersedia setiap 500ms (0.5 detik)
ANNOTATION_INTERVAL = 500