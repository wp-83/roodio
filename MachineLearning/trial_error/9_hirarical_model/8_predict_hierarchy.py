import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import modul fitur (Pastikan file features_hierarchy.py ada di folder yang sama)
import features_hierarchy as feat

# --- KONFIGURASI ---
MODELS_DIR = 'models_hierarchy'
# Gunakan data Test Asli (bukan yang sudah dipisah folder hierarchy, biar fair)
TEST_DIR = 'data/processed_exp4/test' 

print("⏳ Loading 3 Models & Classes...")

# 1. LOAD MODEL STAGE 1 (Arousal)
m1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_1_arousal.h5'))
c1 = np.load(os.path.join(MODELS_DIR, 'classes_stage_1_arousal.npy'), allow_pickle=True)

# 2. LOAD MODEL STAGE 2A (High Group)
m2a = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2a_high_group.h5'))
c2a = np.load(os.path.join(MODELS_DIR, 'classes_stage_2a_high_group.npy'), allow_pickle=True)

# 3. LOAD MODEL STAGE 2B (Low Group)
m2b = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2b_low_group.h5'))
c2b = np.load(os.path.join(MODELS_DIR, 'classes_stage_2b_low_group.npy'), allow_pickle=True)

print("✅ Models Loaded Successfully!")

def predict_hierarchical(file_path):
    """
    Fungsi Pintar: Menentukan emosi berdasarkan keputusan bertingkat
    """
    # --- LEVEL 1: HIGH vs LOW ENERGY ---
    vec1 = feat.extract_stage_1(file_path).reshape(1, -1)
    p1 = m1.predict(vec1, verbose=0)[0]
    stage1_pred = c1[np.argmax(p1)] # Output: 'high' atau 'low'
    
    final_pred = ""
    
    if stage1_pred == 'high':
        # --- LEVEL 2A: ANGRY vs HAPPY ---
        vec2a = feat.extract_stage_2a(file_path).reshape(1, -1)
        p2a = m2a.predict(vec2a, verbose=0)[0]
        final_pred = c2a[np.argmax(p2a)] # Output: 'angry' atau 'happy'
        
    else: # Jika low
        # --- LEVEL 2B: SAD vs RELAXED ---
        vec2b = feat.extract_stage_2b(file_path).reshape(1, -1)
        p2b = m2b.predict(vec2b, verbose=0)[0]
        final_pred = c2b[np.argmax(p2b)] # Output: 'sad' atau 'relaxed'
        
    return final_pred

def evaluate_test_set():
    y_true = []
    y_pred = []
    
    # Urutan label untuk Matrix
    moods = ['angry', 'happy', 'sad', 'relaxed']
    
    print(f"\n🚀 Memulai Evaluasi Hierarki pada Test Set: {TEST_DIR}")
    
    for mood in moods:
        folder = os.path.join(TEST_DIR, mood)
        if not os.path.exists(folder): 
            print(f"⚠️ Warning: Folder {mood} tidak ditemukan di test set.")
            continue
        
        files = os.listdir(folder)
        # Loop setiap file di folder test
        for f in tqdm(files, desc=f"Testing {mood.upper()}"):
            if not f.endswith(('.wav', '.mp3')): continue
            
            try:
                file_path = os.path.join(folder, f)
                
                # PREDIKSI HIERARKI
                prediction = predict_hierarchical(file_path)
                
                y_true.append(mood)
                y_pred.append(prediction)
            except Exception as e:
                print(f"Error {f}: {e}")

    # --- TAMPILKAN HASIL ---
    print("\n" + "="*50)
    print("🏆 FINAL RESULTS EXPERIMENT 8 (HIERARCHY)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=moods))
    
    # --- PLOT CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred, labels=moods)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=moods, yticklabels=moods, annot_kws={"size": 14})
    
    plt.title('Confusion Matrix: Hierarchical Model (Thayer Logic)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    save_path = 'cm_exp8_hierarchy_final.png'
    plt.savefig(save_path)
    print(f"\n✅ Confusion Matrix disimpan ke: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_test_set()