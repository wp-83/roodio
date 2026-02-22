import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
from scipy import signal
import random
import warnings

# Abaikan warning librosa yang tidak kritikal
warnings.filterwarnings("ignore")

# --- KONFIGURASI ---
INPUT_DIR = 'data/split_exp4'  # Menggunakan split data yang sudah bersih
OUTPUT_DIR = 'data/processed_exp5_mood_aware' # Output khusus Exp 5
TARGET_SR = 16000

# Konfigurasi augmentasi berdasarkan mood (Thayer's Arousal-Valence Model)
MOOD_AUGMENTATION_RULES = {
    'happy': {
        'valence': 'positive',
        'arousal': 'high',
        'allowed_augmentations': [
            'white_noise', 'colored_noise', 'time_shift',
            'pitch_shift_small', 'time_stretch_small',
            'random_gain', 'bandpass_filter'
        ],
        'pitch_range': (-1, 1),
        'time_stretch_range': (0.95, 1.05),
        'gain_range': (-4, 6),
        'max_augmentations': 2
    },
    'sad': {
        'valence': 'negative',
        'arousal': 'low',
        'allowed_augmentations': [
            'white_noise', 'colored_noise', 'time_shift',
            'pitch_shift_down', 'time_stretch_slow',
            'random_gain', 'bandpass_filter_low'
        ],
        'pitch_range': (-2, 0),  # Pitch lebih rendah (sedih)
        'time_stretch_range': (0.9, 1.0), # Lebih lambat
        'gain_range': (-8, 2),
        'max_augmentations': 2
    },
    'angry': {
        'valence': 'negative',
        'arousal': 'high',
        'allowed_augmentations': [
            'white_noise', 'colored_noise', 'time_shift',
            'pitch_shift_small', 'time_stretch_fast',
            'random_gain', 'bandpass_filter',
            'harmonic_distortion_mild', 'compression'
        ],
        'pitch_range': (-0.5, 1.5),
        'time_stretch_range': (1.0, 1.1), # Lebih cepat (agresif)
        'gain_range': (0, 8),
        'distortion_level': (0.1, 0.3),
        'max_augmentations': 3
    },
    'relaxed': {
        'valence': 'positive',
        'arousal': 'low',
        'allowed_augmentations': [
            'white_noise_very_mild', 'colored_noise',
            'time_shift', 'pitch_shift_very_small',
            'time_stretch_slow', 'random_gain_quiet',
            'bandpass_filter_smooth'
        ],
        'pitch_range': (-0.5, 0.5),
        'time_stretch_range': (0.9, 1.0),
        'gain_range': (-10, 0),
        'max_augmentations': 1 # Jangan terlalu banyak ubah data relaxed
    }
}

def trim_middle(y, sr, percentage=0.5):
    if len(y) == 0: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

# --- FUNGSI AUGMENTASI ---
def add_white_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def add_colored_noise(y, noise_level=0.003):
    # Simulasi Pink Noise sederhana (kumulatif sum)
    pink_noise = np.cumsum(np.random.randn(len(y)))
    # Normalize pink noise
    if np.max(np.abs(pink_noise)) > 0:
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
    return y + pink_noise * noise_level

def time_shift(y, sr, max_shift_seconds=0.5):
    shift = np.random.randint(-int(sr * max_shift_seconds), int(sr * max_shift_seconds))
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def random_gain(y, db_range=(-6, 6)):
    db = np.random.uniform(db_range[0], db_range[1])
    gain = 10 ** (db / 20)
    return y * gain

def bandpass_filter(y, sr, lowcut=80, highcut=8000):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    # Ensure valid range
    low = max(0.001, min(0.99, low))
    high = max(0.001, min(0.99, high))
    if low >= high: return y
    
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, y)

def harmonic_distortion(y, drive=0.1):
    return np.tanh(y * drive)

def dynamic_range_compression(y, threshold=0.5, ratio=2):
    compressed = np.copy(y)
    mask = np.abs(y) > threshold
    compressed[mask] = threshold + (compressed[mask] - threshold) / ratio
    return compressed

# --- IMPLEMENTASI LOGIKA MOOD-AWARE ---
def apply_mood_aware_augmentation(y, sr, mood):
    rules = MOOD_AUGMENTATION_RULES[mood]
    augmented = y.copy()
    
    # Pilih augmentasi secara random dari daftar yang diizinkan
    num_aug = random.randint(1, rules['max_augmentations'])
    selected_augs = random.sample(rules['allowed_augmentations'], min(num_aug, len(rules['allowed_augmentations'])))
    
    for aug_type in selected_augs:
        try:
            if aug_type == 'white_noise':
                augmented = add_white_noise(augmented, noise_level=random.uniform(0.002, 0.008))
            elif aug_type == 'white_noise_very_mild':
                augmented = add_white_noise(augmented, noise_level=random.uniform(0.001, 0.003))
            elif aug_type == 'colored_noise':
                augmented = add_colored_noise(augmented, noise_level=random.uniform(0.001, 0.004))
            elif aug_type == 'time_shift':
                max_shift = 0.3 if mood == 'relaxed' else random.uniform(0.3, 1.0)
                augmented = time_shift(augmented, sr, max_shift)
            elif aug_type == 'pitch_shift_small':
                steps = random.uniform(rules['pitch_range'][0], rules['pitch_range'][1])
                augmented = pitch_shift(augmented, sr, steps)
            elif aug_type == 'pitch_shift_down':
                steps = random.uniform(-2, 0)
                augmented = pitch_shift(augmented, sr, steps)
            elif aug_type == 'pitch_shift_very_small':
                steps = random.uniform(-0.5, 0.5)
                augmented = pitch_shift(augmented, sr, steps)
            elif aug_type == 'time_stretch_small':
                rate = random.uniform(rules['time_stretch_range'][0], rules['time_stretch_range'][1])
                augmented = time_stretch(augmented, rate)
            elif aug_type == 'time_stretch_slow':
                rate = random.uniform(0.85, 1.0)
                augmented = time_stretch(augmented, rate)
            elif aug_type == 'time_stretch_fast':
                rate = random.uniform(1.0, 1.15)
                augmented = time_stretch(augmented, rate)
            elif aug_type == 'random_gain':
                augmented = random_gain(augmented, rules['gain_range'])
            elif aug_type == 'random_gain_quiet':
                augmented = random_gain(augmented, (-10, 0))
            elif aug_type == 'bandpass_filter':
                augmented = bandpass_filter(augmented, sr, random.randint(50, 200), random.randint(6000, 12000))
            elif aug_type == 'bandpass_filter_low':
                augmented = bandpass_filter(augmented, sr, random.randint(50, 100), random.randint(4000, 8000))
            elif aug_type == 'bandpass_filter_smooth':
                augmented = bandpass_filter(augmented, sr, random.randint(60, 150), random.randint(8000, 14000))
            elif aug_type == 'harmonic_distortion_mild':
                drive = random.uniform(0.05, 0.2)
                augmented = harmonic_distortion(augmented, drive)
            elif aug_type == 'compression':
                augmented = dynamic_range_compression(augmented, random.uniform(0.4, 0.7), random.uniform(2, 4))
        except Exception as e:
            pass # Skip jika augmentasi gagal (misal audio terlalu pendek)

    # Normalize final output
    if np.max(np.abs(augmented)) > 0:
        augmented = augmented / np.max(np.abs(augmented)) * 0.95
    
    return augmented

def augment_per_class(mood, original_count):
    # Adaptive Augmentation: Kelas dengan data sedikit diperbanyak lebih agresif
    if original_count < 100: return 6 # Data sangat sedikit -> x6
    elif original_count < 200: return 4 # Data sedang -> x4
    else: return 2 # Data banyak -> x2

def process_dataset():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    # 1. Hitung distribusi kelas dulu
    class_counts = {}
    for mood in MOOD_AUGMENTATION_RULES.keys():
        src_path = os.path.join(INPUT_DIR, 'train', mood)
        if os.path.exists(src_path):
            class_counts[mood] = len([f for f in os.listdir(src_path) if f.endswith(('.wav', '.mp3'))])
    
    print("\n📊 Class Distribution Analysis (Train):")
    for mood, count in class_counts.items():
        print(f"  {mood}: {count} samples")

    # 2. Proses Data
    for subset in ['train', 'test']:
        for mood in MOOD_AUGMENTATION_RULES.keys():
            src_path = os.path.join(INPUT_DIR, subset, mood)
            dst_path = os.path.join(OUTPUT_DIR, subset, mood)
            os.makedirs(dst_path, exist_ok=True)
            
            if not os.path.exists(src_path): continue
            
            files = [f for f in os.listdir(src_path) if f.endswith(('.wav', '.mp3'))]
            print(f"\n🎵 Processing {subset.upper()} - {mood}...")
            
            for f in tqdm(files, desc=f"{mood}", leave=False):
                try:
                    y, sr = librosa.load(os.path.join(src_path, f), sr=TARGET_SR)
                    y = trim_middle(y, sr, 0.5)
                    
                    base_name = os.path.splitext(f)[0]
                    
                    # Simpan Original
                    sf.write(os.path.join(dst_path, f"{base_name}.wav"), y, sr)
                    
                    # Augmentasi (HANYA TRAIN)
                    if subset == 'train':
                        aug_factor = augment_per_class(mood, class_counts.get(mood, 0))
                        for i in range(aug_factor):
                            aug_y = apply_mood_aware_augmentation(y, sr, mood)
                            sf.write(os.path.join(dst_path, f"{base_name}_aug{i}.wav"), aug_y, sr)
                            
                except Exception as e:
                    print(f"⚠️ Error {f}: {e}")

def generate_report():
    print("\n" + "="*60)
    print("✅ EXPERIMENT 5 DATASET READY")
    print("="*60)
    
    total_files = 0
    for subset in ['train', 'test']:
        print(f"\n{subset.upper()} SET:")
        for mood in MOOD_AUGMENTATION_RULES.keys():
            path = os.path.join(OUTPUT_DIR, subset, mood)
            if os.path.exists(path):
                count = len(os.listdir(path))
                total_files += count
                print(f"  {mood}: {count} files")
    
    print(f"\n📈 Total Dataset Size: {total_files} samples")
    print(f"📂 Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("🚀 STARTING DEEP MOOD-AWARE AUGMENTATION...")
    process_dataset()
    generate_report()