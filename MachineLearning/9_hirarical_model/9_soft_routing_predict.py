import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import feature extractor
import features_hierarchy as feat

# --- KONFIGURASI ---
MODELS_DIR = 'models_hierarchy'
TEST_DIR = 'data/processed_exp4/test' 
THRESHOLD_LOWER = 0.35  # Batas Bawah keraguan (Bisa diset 0.4)
THRESHOLD_UPPER = 0.65  # Batas Atas keraguan (Bisa diset 0.6)

print("⏳ Loading Models...")
# Load Model Stage 1
m1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_1_arousal.h5'))
c1 = np.load(os.path.join(MODELS_DIR, 'classes_stage_1_arousal.npy'), allow_pickle=True)
# Cari index mana yang 'high', mana yang 'low'
idx_high = np.where(c1 == 'high')[0][0]
idx_low  = np.where(c1 == 'low')[0][0]

# Load Model Stage 2A (High)
m2a = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2a_high_group.h5'))
c2a = np.load(os.path.join(MODELS_DIR, 'classes_stage_2a_high_group.npy'), allow_pickle=True)

# Load Model Stage 2B (Low)
m2b = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2b_low_group.h5'))
c2b = np.load(os.path.join(MODELS_DIR, 'classes_stage_2b_low_group.npy'), allow_pickle=True)

print("✅ Models Loaded! Soft Routing Active.")

def predict_soft_routing(file_path):
    # --- STAGE 1: CEK ENERGY ---
    vec1 = feat.extract_stage_1(file_path).reshape(1, -1)
    prob_s1 = m1.predict(vec1, verbose=0)[0] # Contoh: [0.45, 0.55]
    
    # Ambil probabilitas kelas 'high'
    score_high = prob_s1[idx_high]
    
    final_pred = ""
    route_status = "" # Buat debug
    
    # --- LOGIKA SOFT ROUTING ---
    # Jika score_high ada di area abu-abu (misal 0.4 s/d 0.6), kita bingung.
    if THRESHOLD_LOWER <= score_high <= THRESHOLD_UPPER:
        route_status = "AMBIGUOUS -> DUAL ROUTING"
        
        # 1. Tanya Model High (Angry/Happy)
        vec2a = feat.extract_stage_2a(file_path).reshape(1, -1)
        prob_2a = m2a.predict(vec2a, verbose=0)[0]
        conf_2a = np.max(prob_2a)      # Seberapa yakin dia?
        pred_2a = c2a[np.argmax(prob_2a)]
        
        # 2. Tanya Model Low (Sad/Relaxed)
        vec2b = feat.extract_stage_2b(file_path).reshape(1, -1)
        prob_2b = m2b.predict(vec2b, verbose=0)[0]
        conf_2b = np.max(prob_2b)      # Seberapa yakin dia?
        pred_2b = c2b[np.argmax(prob_2b)]
        
        # 3. BATTLE OF CONFIDENCE
        # Siapa yang lebih PD (Percaya Diri), dia yang menang.
        if conf_2a > conf_2b:
            final_pred = pred_2a
        else:
            final_pred = pred_2b
            
    # --- HARD ROUTING (Jika Yakin Banget) ---
    elif score_high > THRESHOLD_UPPER:
        # Yakin High -> Kirim ke 2A
        route_status = "CONFIDENT HIGH"
        vec2a = feat.extract_stage_2a(file_path).reshape(1, -1)
        p2a = m2a.predict(vec2a, verbose=0)[0]
        final_pred = c2a[np.argmax(p2a)]
        
    else: # score_high < THRESHOLD_LOWER
        # Yakin Low -> Kirim ke 2B
        route_status = "CONFIDENT LOW"
        vec2b = feat.extract_stage_2b(file_path).reshape(1, -1)
        p2b = m2b.predict(vec2b, verbose=0)[0]
        final_pred = c2b[np.argmax(p2b)]
        
    return final_pred, route_status

def evaluate_test_set():
    y_true = []
    y_pred = []
    ambiguous_count = 0
    
    moods = ['angry', 'happy', 'sad', 'relaxed']
    
    print(f"\n🚀 Memulai Evaluasi SOFT ROUTING pada: {TEST_DIR}")
    print(f"⚖️  Threshold Ambigu: {THRESHOLD_LOWER} s/d {THRESHOLD_UPPER}")
    
    for mood in moods:
        folder = os.path.join(TEST_DIR, mood)
        if not os.path.exists(folder): continue
        
        files = os.listdir(folder)
        for f in tqdm(files, desc=f"Testing {mood.upper()}"):
            if not f.endswith(('.wav', '.mp3')): continue
            
            try:
                file_path = os.path.join(folder, f)
                prediction, status = predict_soft_routing(file_path)
                
                if "AMBIGUOUS" in status:
                    ambiguous_count += 1
                
                y_true.append(mood)
                y_pred.append(prediction)
            except Exception as e:
                print(f"Error {f}: {e}")

    print("\n" + "="*50)
    print(f"🏆 FINAL RESULT (SOFT ROUTING)")
    print(f"🔍 Total Kasus Ambigu yang diselamatkan: {ambiguous_count}")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=moods))
    
    # Plot CM
    cm = confusion_matrix(y_true, y_pred, labels=moods)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=moods, yticklabels=moods, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix: Soft Routing (Thresh {THRESHOLD_LOWER}-{THRESHOLD_UPPER})', fontsize=16)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('cm_exp9_soft_routing.png')
    plt.show()

if __name__ == "__main__":
    evaluate_test_set()