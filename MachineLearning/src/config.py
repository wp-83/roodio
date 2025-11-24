import os

# --- PATHS ---
# Mendapatkan path absolut agar tidak error saat dijalankan dari manapun
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw", "audio")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "metadata", "audio")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "audio")
MODELS_DIR = os.path.join(BASE_DIR, "models") 
METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata", "lyrics")

# --- AUDIO SETTINGS ---
SAMPLE_RATE = 22050   
DURATION = 45         
SKIP_SECONDS = 15     
SEGMENT_LENGTH = 3    

# --- SPECTROGRAM SETTINGS (PENTING: Jangan Diubah) ---
N_MELS = 128        
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 8000