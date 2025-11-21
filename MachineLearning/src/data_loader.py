# src/data_loader.py
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import src.config as cfg  # Import config yang baru kita buat

class DEAMDataLoader:
    def __init__(self):
        self.valence_path = os.path.join(cfg.ANNOTATIONS_DIR, "valence.csv")
        self.arousal_path = os.path.join(cfg.ANNOTATIONS_DIR, "arousal.csv")
        
    def _load_and_clean_labels(self):
        """
        Internal function: Menggabungkan dan membersihkan CSV Valence & Arousal
        (Logika sama persis dengan yang Anda temukan di EDA)
        """
        def melt_df(path, label_name):
            df = pd.read_csv(path)
            df_melted = df.melt(id_vars=['song_id'], var_name='timestamp_str', value_name=label_name)
            # Ubah 'sample_15000ms' menjadi integer 15000
            df_melted['timestamp'] = df_melted['timestamp_str'].str.extract('(\d+)').astype(int)
            return df_melted.drop(columns=['timestamp_str'])

        print("   > Loading CSVs...")
        df_v = melt_df(self.valence_path, 'valence')
        df_a = melt_df(self.arousal_path, 'arousal')
        
        # Inner Join untuk sinkronisasi
        df_final = pd.merge(df_v, df_a, on=['song_id', 'timestamp'], how='inner')
        
        # Hapus NaN
        df_final = df_final.dropna()
        
        # Pastikan hanya mengambil data > 15000ms
        df_final = df_final[df_final['timestamp'] >= (cfg.SKIP_SECONDS * 1000)]
        
        return df_final

    def _extract_spectrogram(self, y):
        """Mengubah audio raw menjadi Mel-Spectrogram"""
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=cfg.SAMPLE_RATE, 
            n_mels=cfg.N_MELS, 
            n_fft=cfg.N_FFT, 
            hop_length=cfg.HOP_LENGTH, 
            fmax=cfg.FMAX
        )
        # Ubah ke Log-Scale (dB)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Normalisasi Min-Max (agar range 0 s.d 1)
        # Tambahkan epsilon kecil agar tidak divide by zero jika file hening
        min_val = S_dB.min()
        max_val = S_dB.max()
        if max_val - min_val == 0:
            return np.zeros_like(S_dB, dtype=np.float32)
        
        S_norm = (S_dB - min_val) / (max_val - min_val)
        return S_norm.astype(np.float32)

    def process_and_save(self):
        """Fungsi Utama: Loop audio, potong, ekstrak fitur, simpan."""
        
        # 1. Siapkan Label
        print("1. Memproses Anotasi...")
        df_labels = self._load_and_clean_labels()
        
        # Siapkan list penampung
        X_data = [] # Gambar Spectrogram
        y_data = [] # Label [Valence, Arousal]
        
        # Ambil list lagu unik
        unique_songs = df_labels['song_id'].unique()
        
        # Buat folder output jika belum ada
        if not os.path.exists(cfg.PROCESSED_DATA_DIR):
            os.makedirs(cfg.PROCESSED_DATA_DIR)
            
        print(f"2. Memproses Audio ({len(unique_songs)} lagu)...")
        
        # Loop setiap lagu (Gunakan tqdm untuk progress bar)
        for song_id in tqdm(unique_songs):
            audio_path = os.path.join(cfg.RAW_AUDIO_DIR, f"{song_id}.mp3")
            
            if not os.path.exists(audio_path):
                continue
                
            try:
                # Load Audio (Mulai dari detik ke-15 sampai 45)
                # Duration 30 detik (45 - 15)
                load_duration = cfg.DURATION - cfg.SKIP_SECONDS
                y, _ = librosa.load(audio_path, sr=cfg.SAMPLE_RATE, offset=cfg.SKIP_SECONDS, duration=load_duration)
                
                # Total sampel yang dibutuhkan untuk 3 detik
                samples_per_segment = cfg.SEGMENT_LENGTH * cfg.SAMPLE_RATE
                total_segments = len(y) // samples_per_segment
                
                # Loop Segmentasi (Potong-potong)
                for i in range(total_segments):
                    start_idx = i * samples_per_segment
                    end_idx = start_idx + samples_per_segment
                    
                    segment_audio = y[start_idx:end_idx]
                    
                    # Pastikan panjang segmen pas
                    if len(segment_audio) < samples_per_segment:
                        continue
                        
                    # --- A. PROSES AUDIO ---
                    spec = self._extract_spectrogram(segment_audio)
                    
                    # Fix shape ke 128x128 (kadang outputnya 128x130)
                    if spec.shape[1] > 128:
                        spec = spec[:, :128]
                    elif spec.shape[1] < 128:
                        # Padding jika kurang
                        pad_width = 128 - spec.shape[1]
                        spec = np.pad(spec, ((0,0), (0,pad_width)))
                    
                    # --- B. PROSES LABEL ---
                    # Hitung waktu absolut segmen ini dalam ms
                    # Start = 15000 + (i * 3000)
                    segment_start_ms = (cfg.SKIP_SECONDS * 1000) + (i * cfg.SEGMENT_LENGTH * 1000)
                    segment_end_ms = segment_start_ms + (cfg.SEGMENT_LENGTH * 1000)
                    
                    # Ambil label yang berada dalam rentang waktu segmen ini
                    mask = (df_labels['song_id'] == song_id) & \
                           (df_labels['timestamp'] >= segment_start_ms) & \
                           (df_labels['timestamp'] < segment_end_ms)
                    
                    segment_labels = df_labels[mask]
                    
                    if segment_labels.empty:
                        continue
                        
                    # Ambil Rata-rata Valence & Arousal
                    avg_v = segment_labels['valence'].mean()
                    avg_a = segment_labels['arousal'].mean()
                    
                    # Simpan ke list
                    X_data.append(spec)
                    y_data.append([avg_v, avg_a])
                    
            except Exception as e:
                print(f"Error pada lagu {song_id}: {e}")
                continue

        # 3. Convert ke NumPy Array & Simpan
        print("3. Menyimpan Dataset...")
        X_array = np.array(X_data) # Shape: (N, 128, 128)
        y_array = np.array(y_data) # Shape: (N, 2)
        
        # Tambah dimensi channel untuk CNN (N, 128, 128, 1)
        X_array = X_array[..., np.newaxis]
        
        np.save(os.path.join(cfg.PROCESSED_DATA_DIR, "X_train.npy"), X_array)
        np.save(os.path.join(cfg.PROCESSED_DATA_DIR, "y_train.npy"), y_array)
        
        print(f"✅ Selesai! Dataset tersimpan di {cfg.PROCESSED_DATA_DIR}")
        print(f"Shape X: {X_array.shape}")
        print(f"Shape y: {y_array.shape}")

if __name__ == "__main__":
    loader = DEAMDataLoader()
    loader.process_and_save()