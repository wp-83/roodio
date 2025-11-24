import streamlit as st
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import modul lokal
import config as cfg
from lyrics_analyzer import NRCLexiconAnalyzer

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Moodio Multimodal", page_icon="🎹", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # 1. Audio Model
    audio_path = os.path.join(cfg.MODELS_DIR, "model_emosi_cnn.keras")
    if not os.path.exists(audio_path): return None, None
    audio_model = load_model(audio_path)
    
    # 2. Lyrics Model
    lyrics_model = NRCLexiconAnalyzer() 
    
    return audio_model, lyrics_model

audio_model, lyrics_model = load_models()

if audio_model is None:
    st.error("Model Audio tidak ditemukan! Pastikan file 'models/model_emosi_cnn.keras' ada.")
    st.stop()

# --- FUNGSI UTAMA ---
def normalize_value(val):
    """
    Normalisasi output Audio (Range -0.8 s.d 0.8) ke skala 0.0 - 1.0.
    Menggunakan range kalibrasi 0.05 s.d 0.45 agar sensitif terhadap perubahan kecil.
    """
    MIN_VAL = 0.05
    MAX_VAL = 0.45
    norm = (val - MIN_VAL) / (MAX_VAL - MIN_VAL)
    return np.clip(norm, 0.0, 1.0)

def preprocess_spectrogram(y):
    S = librosa.feature.melspectrogram(y=y, sr=cfg.SAMPLE_RATE, n_mels=cfg.N_MELS, 
                                      n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, fmax=cfg.FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)
    min_val, max_val = S_dB.min(), S_dB.max()
    if max_val - min_val == 0: S_norm = np.zeros_like(S_dB)
    else: S_norm = (S_dB - min_val) / (max_val - min_val)
    
    if S_norm.shape[1] < 128:
        pad = 128 - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0,0), (0,pad)))
    else:
        S_norm = S_norm[:, :128]
    return S_norm[np.newaxis, ..., np.newaxis]

def get_mood_label(v, a):
    if a >= 0.5:
        return ("Happy / Energetic", "orange") if v >= 0.5 else ("Angry / Tense", "red")
    else:
        return ("Calm / Peaceful", "green") if v >= 0.5 else ("Sad / Melancholy", "blue")

def plot_quadrant(v_final, a_final, v_audio, a_audio, v_lyrics, a_lyrics):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    
    # Label Kuadran
    ax.text(0.95, 0.95, "HAPPY", color='orange', ha='right', weight='bold', alpha=0.5)
    ax.text(0.05, 0.95, "ANGRY", color='red', ha='left', weight='bold', alpha=0.5)
    ax.text(0.05, 0.05, "SAD", color='blue', ha='left', weight='bold', alpha=0.5)
    ax.text(0.95, 0.05, "CALM", color='green', ha='right', weight='bold', alpha=0.5)
    
    # Plot Points
    ax.scatter(v_audio, a_audio, c='blue', marker='s', s=100, label='Audio', alpha=0.6)
    ax.scatter(v_lyrics, a_lyrics, c='green', marker='^', s=100, label='Lirik', alpha=0.6)
    ax.scatter(v_final, a_final, c='purple', s=300, label='Final', edgecolors='black', zorder=10)
    
    # Garis Konektor
    ax.plot([v_audio, v_final], [a_audio, a_final], 'k--', alpha=0.2)
    ax.plot([v_lyrics, v_final], [a_lyrics, a_final], 'k--', alpha=0.2)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax.set_title("Peta Emosi Multimodal")
    return fig

# --- UI STREAMLIT ---
st.title("🎹 Moodio: Multimodal Emotion AI")
st.markdown("Analisis emosi lagu menggunakan gabungan **Audio (CNN)** dan **Lirik (Lexicon)** dengan pembobotan dinamis.")

# --- SIDEBAR: KONFIGURASI BOBOT (DYNAMIC WEIGHTING) ---
with st.sidebar:
    st.header("⚙️ Konfigurasi Fusion")
    st.info("Bobot (Weight) menentukan seberapa besar pengaruh Audio vs Lirik terhadap hasil akhir.")
    
    st.subheader("1. Bobot Valence (Positif/Negatif)")
    st.caption("Disarankan Lirik lebih dominan untuk menentukan Happy/Sad.")
    w_val_audio = st.slider("Audio Valence Weight", 0.0, 1.0, 0.4, step=0.1)
    w_val_lyrics = 1.0 - w_val_audio
    st.text(f"Audio: {w_val_audio:.1f} | Lirik: {w_val_lyrics:.1f}")
    
    st.divider()
    
    st.subheader("2. Bobot Arousal (Energi)")
    st.caption("Disarankan Audio lebih dominan untuk menentukan Energi.")
    w_aro_audio = st.slider("Audio Arousal Weight", 0.0, 1.0, 0.7, step=0.1)
    w_aro_lyrics = 1.0 - w_aro_audio
    st.text(f"Audio: {w_aro_audio:.1f} | Lirik: {w_aro_lyrics:.1f}")

# --- MAIN CONTENT ---
c1, c2 = st.columns(2)
with c1:
    uploaded_file = st.file_uploader("1. Upload Audio (MP3/WAV)", type=["mp3", "wav"])
with c2:
    lyrics_input = st.text_area("2. Masukkan Lirik (Inggris)", height=100, placeholder="Paste lyrics here...")

if st.button("🔍 Analisis Emosi", type="primary"):
    if not uploaded_file or not lyrics_input.strip():
        st.warning("Mohon lengkapi Audio dan Lirik.")
    else:
        with st.spinner("Sedang menganalisis..."):
            # --- 1. AUDIO PROCESS ---
            with open("temp.mp3", "wb") as f: f.write(uploaded_file.getbuffer())
            y, sr = librosa.load("temp.mp3", sr=cfg.SAMPLE_RATE, mono=True)
            
            # Sampling Audio
            total_segs = len(y) // (3 * cfg.SAMPLE_RATE)
            indices = np.linspace(0, total_segs-1, num=min(30, total_segs), dtype=int)
            
            v_list, a_list = [], []
            for i in indices:
                start = i * 3 * cfg.SAMPLE_RATE
                segment = y[start : start + (3 * cfg.SAMPLE_RATE)]
                try:
                    spec = preprocess_spectrogram(segment)
                    pred = audio_model.predict(spec, verbose=0)
                    v_list.append(pred[0][0]); a_list.append(pred[0][1])
                except: pass
            
            # Normalize Audio
            v_aud_norm = normalize_value(np.mean(v_list))
            a_aud_norm = normalize_value(np.mean(a_list))
            
            # --- 2. LYRICS PROCESS ---
            # (Sudah 0-1 dari lyrics_analyzer)
            v_lyr_norm, a_lyr_norm, info = lyrics_model.predict(lyrics_input)
            
            # --- 3. FUSION (DYNAMIC) ---
            # Menggunakan bobot terpisah untuk Valence dan Arousal
            final_v = (v_aud_norm * w_val_audio) + (v_lyr_norm * w_val_lyrics)
            final_a = (a_aud_norm * w_aro_audio) + (a_lyr_norm * w_aro_lyrics)
            
            mood_label, mood_color = get_mood_label(final_v, final_a)
            
            # --- 4. DISPLAY ---
            st.divider()
            r1, r2 = st.columns([1.3, 1])
            
            with r1:
                st.markdown(f"### Prediksi: :{mood_color}[{mood_label}]")
                
                if info['keywords']:
                    st.success(f"Kata Kunci Lirik: {', '.join(info['keywords'])}")
                else:
                    st.warning("Tidak ada kata emosi spesifik di lirik.")
                
                # Tabel Hasil Detail
                results_data = {
                    'Valence (Positif)': [f"{v_aud_norm:.2f}", f"{v_lyr_norm:.2f}", f"**{final_v:.2f}**"],
                    'Arousal (Energi)': [f"{a_aud_norm:.2f}", f"{a_lyr_norm:.2f}", f"**{final_a:.2f}**"],
                    'Bobot (V | A)': [f"{w_val_audio:.1f} | {w_aro_audio:.1f}", f"{w_val_lyrics:.1f} | {w_aro_lyrics:.1f}", "-"]
                }
                df_res = pd.DataFrame(results_data, index=['Audio (CNN)', 'Lirik (NRC)', 'FINAL FUSION'])
                st.table(df_res)
                
            with r2:
                st.pyplot(plot_quadrant(final_v, final_a, v_aud_norm, a_aud_norm, v_lyr_norm, a_lyr_norm))
            
            if os.path.exists("temp.mp3"): os.remove("temp.mp3")