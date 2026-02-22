import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- KONFIGURASI PATH (ANTI-TERSESAT) ---
# Menggunakan pathlib agar path selalu akurat relatif terhadap lokasi script ini
CURRENT_FILE = Path(__file__).resolve()
SCRIPT_DIR = CURRENT_FILE.parent       # Folder 'cnn_experiment'
PROJECT_DIR = CURRENT_FILE.parents[1]  # Folder 'machineLearning'

# Path Absolut
DATA_DIR = PROJECT_DIR / 'data' / 'processed_cnn' / 'test'
MODEL_PATH = SCRIPT_DIR / 'models_cnn' / 'audio_mood_cnn.h5'
CLASSES_PATH = SCRIPT_DIR / 'models_cnn' / 'classes.npy'

IMG_WIDTH = 130 

def main():
    print("⚖️ MENGHITUNG AKURASI DENGAN SISTEM VOTING (PER LAGU)...")
    print(f"   📂 Lokasi Script: {SCRIPT_DIR}")
    print(f"   📂 Mencari Model di: {MODEL_PATH}")
    
    # 1. Cek File Dulu
    if not MODEL_PATH.exists():
        print("❌ CRITICAL ERROR: File model (.h5) tidak ditemukan!")
        print("   Pastikan '4_train_cnn.py' sudah dijalankan sampai sukses.")
        return

    # 2. Load Model & Label
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        classes = np.load(CLASSES_PATH)
        print(f"✅ Model berhasil dimuat. Classes: {classes}")
    except Exception as e:
        print(f"❌ Gagal load model (Mungkin corrupt/versi beda). Error: {e}")
        return

    y_true_songs = []
    y_pred_songs = []

    # 3. Loop per Folder Mood (Angry, Happy, etc)
    # Pastikan loop berdasarkan index kelas yang benar
    for mood_idx, mood in enumerate(classes):
        mood_path = DATA_DIR / mood
        if not mood_path.exists(): 
            print(f"⚠️ Warning: Folder test untuk '{mood}' tidak ditemukan di {mood_path}")
            continue

        # Kelompokkan file berdasarkan nama lagu asli
        # Contoh: "LaguA_seg0.npy", "LaguA_seg1.npy" -> Group "LaguA"
        song_groups = {}
        files = [f for f in os.listdir(mood_path) if f.endswith('.npy')]
        
        for f in files:
            # Ambil nama lagu (hapus _segX.npy)
            song_name = f.rsplit('_seg', 1)[0]
            if song_name not in song_groups:
                song_groups[song_name] = []
            song_groups[song_name].append(mood_path / f)
        
        print(f"   📂 Evaluating {mood}: {len(song_groups)} lagu full...")

        # 4. Prediksi per Lagu (Voting)
        for song_name, segments in song_groups.items():
            
            # Load semua segmen lagu tersebut
            batch_images = []
            for seg_path in segments:
                try:
                    spectrogram = np.load(seg_path)
                    
                    # Padding/Trim (Sama seperti training)
                    if spectrogram.shape[1] < IMG_WIDTH:
                        pad_width = IMG_WIDTH - spectrogram.shape[1]
                        spectrogram = np.pad(spectrogram, ((0,0), (0, pad_width)))
                    else:
                        spectrogram = spectrogram[:, :IMG_WIDTH]
                    
                    batch_images.append(spectrogram)
                except Exception as e:
                    print(f"Error reading segment {seg_path}: {e}")

            if not batch_images: continue
            
            # Predict batch sekaligus biar cepat
            X_batch = np.array(batch_images)[..., np.newaxis]
            predictions = model.predict(X_batch, verbose=0)
            pred_indices = np.argmax(predictions, axis=1)
            
            # --- LOGIKA VOTING ---
            # Cari mood yang paling sering muncul (Mode)
            vote_counts = Counter(pred_indices)
            
            # Ambil juara 1. Jika seri, ambil yang pertama.
            most_common_idx = vote_counts.most_common(1)[0][0] 
            
            y_true_songs.append(mood_idx) # Label asli (dari folder)
            y_pred_songs.append(most_common_idx) # Hasil voting

    # 5. Tampilkan Hasil Akhir
    print("\n" + "="*50)
    print("   HASIL AKHIR (LEVEL LAGU UTUH - SETELAH VOTING)")
    print("="*50)
    
    if len(y_true_songs) == 0:
        print("❌ Tidak ada data lagu yang berhasil diproses.")
        return

    acc = accuracy_score(y_true_songs, y_pred_songs)
    print(f"🏆 REAL SONG ACCURACY: {acc*100:.2f}%")
    print("-" * 50)
    print(classification_report(y_true_songs, y_pred_songs, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_songs, y_pred_songs)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Voting per Song) - Acc: {acc:.2f}')
    plt.xlabel('Prediksi AI')
    plt.ylabel('Label Asli')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()