import streamlit as st
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import config as cfg # Import langsung karena satu folder

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Moodio AI", page_icon="🎵", layout="wide")

# --- FUNGSI HELPER ---
@st.cache_resource
def load_audio_model():
    model_path = os.path.join(cfg.BASE_DIR, "models", "model_emosi_cnn.keras")
    return load_model(model_path)

def normalize_audio(val):
    # Kalibrasi output audio ke skala 0-1
    return np.clip((val + 1) / 2, 0.0, 1.0)

def preprocess_spectrogram(y):
    S = librosa.feature.melspectrogram(y=y, sr=cfg.SAMPLE_RATE, n_mels=cfg.N_MELS, 
                                      n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, fmax=cfg.FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)
    min_val, max_val = S_dB.min(), S_dB.max()
    if max_val - min_val == 0: return np.zeros_like(S_dB)
    S_norm = (S_dB - min_val) / (max_val - min_val)
    
    if S_norm.shape[1] < 128:
        pad = 128 - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0,0), (0,pad)))
    else:
        S_norm = S_norm[:, :128]
        
    return S_norm[np.newaxis, ..., np.newaxis]

def get_mood_label(v, a):
    # Threshold di 0.5
    if a >= 0.5:
        return ("Happy / Energetic", "orange") if v >= 0.5 else ("Angry / Tense", "red")
    else:
        return ("Calm / Peaceful", "green") if v >= 0.5 else ("Sad / Melancholy", "blue")

# --- VISUALISASI 1: KUADRAN (RATA-RATA) ---
def plot_quadrant(valence, arousal):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Label
    ax.text(0.95, 0.95, "HAPPY", color='orange', ha='right', weight='bold')
    ax.text(0.05, 0.95, "ANGRY", color='red', ha='left', weight='bold')
    ax.text(0.05, 0.05, "SAD", color='blue', ha='left', weight='bold')
    ax.text(0.95, 0.05, "CALM", color='green', ha='right', weight='bold')
    
    # Titik
    ax.scatter(valence, arousal, color='purple', s=200, zorder=5, edgecolors='white')
    ax.set_title("Rata-rata Emosi Lagu")
    return fig

# --- VISUALISASI 2: TIMELINE (EMOSI SEPANJANG LAGU) ---
def plot_timeline(v_list, a_list):
    fig, ax = plt.subplots(figsize=(10, 3))
    time_axis = np.arange(len(v_list)) * 3  # Dikali 3 karena per segmen 3 detik
    
    ax.plot(time_axis, v_list, label='Valence (Positivity)', color='blue', linewidth=2)
    ax.plot(time_axis, a_list, label='Arousal (Energy)', color='orange', linewidth=2)
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Waktu (Detik)")
    ax.set_title("Perubahan Emosi Sepanjang Lagu")
    ax.legend()
    ax.grid(True, alpha=0.2)
    return fig

# --- UI UTAMA ---
st.title("🎵 Moodio: AI Music Emotion Recognition")
st.markdown("Upload lagu, dan AI akan menganalisis **setiap detik** dari lagu tersebut.")

try:
    model = load_audio_model()
except:
    st.error("Model tidak ditemukan!")
    st.stop()

uploaded_file = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Analisis Full Lagu"):
        with st.spinner('Sedang memproses seluruh lagu... (Mohon tunggu)'):
            
            # 1. Save Temp
            with open("temp_audio.mp3", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Load
            y, sr = librosa.load("temp_audio.mp3", sr=cfg.SAMPLE_RATE, mono=True)
            
            # 3. Full Loop (Tanpa Limit)
            samples_per_segment = 3 * cfg.SAMPLE_RATE
            total_segments = len(y) // samples_per_segment
            
            v_scores, a_scores = [], []
            progress_bar = st.progress(0)
            
            # Loop SEMUA segmen
            for i in range(total_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = y[start:end]
                
                spec = preprocess_spectrogram(segment)
                pred = model.predict(spec, verbose=0)
                
                v_scores.append(normalize_audio(pred[0][0]))
                a_scores.append(normalize_audio(pred[0][1]))
                
                # Update progress setiap 10% agar tidak memperlambat UI
                if i % max(1, int(total_segments/10)) == 0:
                    progress_bar.progress((i + 1) / total_segments)
            
            progress_bar.progress(1.0) # Selesai
            
            # 4. Hasil Rata-rata
            avg_v = np.mean(v_scores)
            avg_a = np.mean(a_scores)
            label, color = get_mood_label(avg_v, avg_a)
            
            # --- TAMPILAN HASIL ---
            st.divider()
            
            # Bagian Atas: Metrik & Kuadran
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Hasil Analisis")
                st.markdown(f"### Mood Dominan: :{color}[{label}]")
                
                c1, c2 = st.columns(2)
                c1.metric("Rata-rata Valence", f"{avg_v:.2f}")
                c2.metric("Rata-rata Arousal", f"{avg_a:.2f}")
                
                st.info(f"Total durasi yang dianalisis: {total_segments * 3} detik ({total_segments} segmen).")

            with col2:
                fig_quad = plot_quadrant(avg_v, avg_a)
                st.pyplot(fig_quad)
            
            # Bagian Bawah: Grafik Timeline (Fitur Baru!)
            st.subheader("Grafik Detik-ke-Detik")
            fig_time = plot_timeline(v_scores, a_scores)
            st.pyplot(fig_time)
            
            # Cleanup
            if os.path.exists("temp_audio.mp3"):
                os.remove("temp_audio.mp3")