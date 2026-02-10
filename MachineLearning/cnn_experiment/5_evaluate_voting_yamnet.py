import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# --- KONFIGURASI ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]

# Path Data Segmen Test
DATA_DIR = PROJECT_DIR / 'data' / 'processed_yamnet_segments' / 'test'
MODEL_DIR = PROJECT_DIR / 'cnn_experiment' / 'models_yamnet'

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000

def load_models():
    print("📥 Loading YAMNet & Classifier...")
    yamnet = hub.load(YAMNET_URL)
    classifier = tf.keras.models.load_model(MODEL_DIR / 'mood_classifier.h5')
    classes = np.load(MODEL_DIR / 'classes.npy')
    return yamnet, classifier, classes

def extract_embedding(yamnet, file_path):
    try:
        wav, _ = librosa.load(file_path, sr=SR)
        # Normalisasi
        max_val = np.max(np.abs(wav))
        if max_val > 0: wav = wav / max_val
        # Padding
        if len(wav) < 15600: wav = np.pad(wav, (0, 15600 - len(wav)))
        
        scores, embeddings, spectrogram = yamnet(wav)
        global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        return global_embedding
    except:
        return None

def main():
    yamnet, classifier, classes = load_models()
    
    y_true_songs = []
    y_pred_songs = []
    
    print("\n🗳️ MULAI EVALUASI VOTING PER LAGU...")
    
    # Loop per Folder Mood (Angry, Happy, etc)
    for mood_idx, mood in enumerate(classes):
        mood_path = DATA_DIR / mood
        if not mood_path.exists(): continue
        
        # 1. Kelompokkan File per Lagu
        # Contoh: "song1_seg0.wav", "song1_seg1.wav" -> milik "song1"
        song_groups = {}
        files = [f for f in os.listdir(mood_path) if f.endswith('.wav')]
        
        for f in files:
            # Trik: Ambil nama sebelum "_seg" terakhir
            # Misal: "Metallica_Enter_Sandman_seg12.wav" -> "Metallica_Enter_Sandman"
            song_name = f.rsplit('_seg', 1)[0]
            if song_name not in song_groups:
                song_groups[song_name] = []
            song_groups[song_name].append(mood_path / f)
            
        print(f"   📂 {mood}: {len(song_groups)} lagu full ({len(files)} segmen)")
        
        # 2. Proses Per Lagu
        for song_name, segments in tqdm(song_groups.items(), leave=False):
            batch_embeddings = []
            
            # Extract feature semua segmen lagu ini
            for seg_path in segments:
                emb = extract_embedding(yamnet, seg_path)
                if emb is not None:
                    batch_embeddings.append(emb)
            
            if not batch_embeddings: continue
            
            # Predict Batch
            batch_embeddings = np.array(batch_embeddings)
            preds = classifier.predict(batch_embeddings, verbose=0)
            pred_indices = np.argmax(preds, axis=1)
            
            # --- VOTING ---
            vote_counts = Counter(pred_indices)
            # Ambil index mood terbanyak (Juara 1)
            winner_idx = vote_counts.most_common(1)[0][0]
            
            y_true_songs.append(mood_idx)
            y_pred_songs.append(winner_idx)

    # 3. Hasil Akhir
    print("\n" + "="*50)
    print("   HASIL AKHIR (REAL SONG ACCURACY)")
    print("="*50)
    
    acc = accuracy_score(y_true_songs, y_pred_songs)
    print(f"🏆 SONG ACCURACY: {acc*100:.2f}%")
    print("-" * 50)
    print(classification_report(y_true_songs, y_pred_songs, target_names=classes))
    
    # Plot CM
    cm = confusion_matrix(y_true_songs, y_pred_songs)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.title(f'Song Level Voting Matrix - Acc: {acc:.2f}')
    plt.xlabel('Prediksi AI')
    plt.ylabel('Label Asli')
    plt.show()

if __name__ == "__main__":
    main()