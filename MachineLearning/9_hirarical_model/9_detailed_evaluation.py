import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import modul fitur
import features_hierarchy as feat

# --- KONFIGURASI ---
MODELS_DIR = 'models_hierarchy'
TEST_DIR = 'data/processed_exp4/test' 

# Load Models
print("⏳ Loading 3 Models...")
m1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_1_arousal.h5'))
c1 = np.load(os.path.join(MODELS_DIR, 'classes_stage_1_arousal.npy'), allow_pickle=True)

m2a = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2a_high_group.h5'))
c2a = np.load(os.path.join(MODELS_DIR, 'classes_stage_2a_high_group.npy'), allow_pickle=True)

m2b = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2b_low_group.h5'))
c2b = np.load(os.path.join(MODELS_DIR, 'classes_stage_2b_low_group.npy'), allow_pickle=True)

def get_true_arousal(mood):
    """Mapping manual: Mood Asli -> Arousal Asli"""
    if mood in ['angry', 'happy']: return 'high'
    if mood in ['sad', 'relaxed']: return 'low'
    return None

def evaluate_all_stages():
    # List Penampung Data
    # 1. Untuk Stage 1 (Arousal)
    y_true_s1, y_pred_s1 = [], []
    
    # 2. Untuk Stage 2A (Hanya Angry & Happy)
    y_true_s2a, y_pred_s2a = [], []
    
    # 3. Untuk Stage 2B (Hanya Sad & Relaxed)
    y_true_s2b, y_pred_s2b = [], []
    
    # 4. Untuk Final System (End-to-End)
    y_true_final, y_pred_final = [], []
    
    moods = ['angry', 'happy', 'sad', 'relaxed']
    
    print(f"🚀 Memulai Detailed Evaluation pada: {TEST_DIR}")
    
    for mood in moods:
        folder = os.path.join(TEST_DIR, mood)
        if not os.path.exists(folder): continue
        
        files = os.listdir(folder)
        for f in tqdm(files, desc=f"Testing {mood.upper()}"):
            if not f.endswith(('.wav', '.mp3')): continue
            file_path = os.path.join(folder, f)
            
            try:
                # --- A. EVALUASI STAGE 1 (AROUSAL) ---
                true_arousal = get_true_arousal(mood)
                
                vec1 = feat.extract_stage_1(file_path).reshape(1, -1)
                p1 = m1.predict(vec1, verbose=0)[0]
                pred_arousal = c1[np.argmax(p1)]
                
                y_true_s1.append(true_arousal)
                y_pred_s1.append(pred_arousal)
                
                # --- B. EVALUASI STAGE 2 (VALENCE) ---
                # Kita cek performa model 2A/2B secara terisolasi (Standalone)
                # Artinya: Seberapa jago Model 2A jika inputnya PASTI Angry/Happy?
                
                if true_arousal == 'high':
                    vec2a = feat.extract_stage_2a(file_path).reshape(1, -1)
                    p2a = m2a.predict(vec2a, verbose=0)[0]
                    pred_s2a = c2a[np.argmax(p2a)]
                    
                    y_true_s2a.append(mood)
                    y_pred_s2a.append(pred_s2a)
                    
                elif true_arousal == 'low':
                    vec2b = feat.extract_stage_2b(file_path).reshape(1, -1)
                    p2b = m2b.predict(vec2b, verbose=0)[0]
                    pred_s2b = c2b[np.argmax(p2b)]
                    
                    y_true_s2b.append(mood)
                    y_pred_s2b.append(pred_s2b)
                
                # --- C. EVALUASI FINAL (PIPELINE) ---
                # Ini hasil real-world (mengikuti prediksi Stage 1)
                final_res = ""
                if pred_arousal == 'high':
                    # Walaupun aslinya sad, kalau stage 1 bilang high, dia masuk sini
                    vec2a = feat.extract_stage_2a(file_path).reshape(1, -1)
                    p2a = m2a.predict(vec2a, verbose=0)[0]
                    final_res = c2a[np.argmax(p2a)]
                else:
                    vec2b = feat.extract_stage_2b(file_path).reshape(1, -1)
                    p2b = m2b.predict(vec2b, verbose=0)[0]
                    final_res = c2b[np.argmax(p2b)]
                
                y_true_final.append(mood)
                y_pred_final.append(final_res)

            except Exception as e:
                print(f"Error {f}: {e}")

    # --- PLOTTING 4 MATRIX ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. Matrix Stage 1 (High vs Low)
    cm1 = confusion_matrix(y_true_s1, y_pred_s1, labels=['high', 'low'])
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0],
                xticklabels=['High', 'Low'], yticklabels=['High', 'Low'])
    axes[0, 0].set_title('STAGE 1: AROUSAL (Energy)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    
    # 2. Matrix Stage 2A (Angry vs Happy)
    cm2a = confusion_matrix(y_true_s2a, y_pred_s2a, labels=['angry', 'happy'])
    sns.heatmap(cm2a, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1],
                xticklabels=['Angry', 'Happy'], yticklabels=['Angry', 'Happy'])
    axes[0, 1].set_title('STAGE 2A: HIGH VALENCE (Standalone)', fontsize=14, fontweight='bold')
    
    # 3. Matrix Stage 2B (Sad vs Relaxed)
    cm2b = confusion_matrix(y_true_s2b, y_pred_s2b, labels=['sad', 'relaxed'])
    sns.heatmap(cm2b, annot=True, fmt='d', cmap='Purples', ax=axes[1, 0],
                xticklabels=['Sad', 'Relaxed'], yticklabels=['Sad', 'Relaxed'])
    axes[1, 0].set_title('STAGE 2B: LOW VALENCE (Standalone)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('True Label')
    
    # 4. Matrix Final (4 Class)
    cm_final = confusion_matrix(y_true_final, y_pred_final, labels=moods)
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=moods, yticklabels=moods)
    axes[1, 1].set_title('FINAL SYSTEM (Pipeline Result)', fontsize=14, fontweight='bold')

    # Save
    save_path = 'cm_exp8_detailed_stages.png'
    plt.savefig(save_path)
    print(f"\n✅ Analisis Lengkap Disimpan: {save_path}")
    print("\n--- CLASSIFICATION REPORT (FINAL) ---")
    print(classification_report(y_true_final, y_pred_final))
    plt.show()

if __name__ == "__main__":
    evaluate_all_stages()