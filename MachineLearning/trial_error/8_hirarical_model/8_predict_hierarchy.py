import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import features_hierarchy as feat

MODELS_DIR = 'models_hierarchy'
TEST_DIR = 'data/processed_exp4/test' # Data Ujian Asli

print("⏳ Loading Models...")
# Load Model & Classes
m1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_1_arousal.h5'))
c1 = np.load(os.path.join(MODELS_DIR, 'classes_stage_1_arousal.npy'), allow_pickle=True)

m2a = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2a_high_group.h5'))
c2a = np.load(os.path.join(MODELS_DIR, 'classes_stage_2a_high_group.npy'), allow_pickle=True)

m2b = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2b_low_group.h5'))
c2b = np.load(os.path.join(MODELS_DIR, 'classes_stage_2b_low_group.npy'), allow_pickle=True)

def predict_single(file_path):
    # --- STAGE 1: HIGH vs LOW ---
    vec1 = feat.extract_stage_1(file_path).reshape(1, -1)
    p1 = m1.predict(vec1, verbose=0)[0]
    stage1_pred = c1[np.argmax(p1)]
    
    final_pred = ""
    
    if stage1_pred == 'high':
        # --- STAGE 2A: ANGRY vs HAPPY ---
        vec2a = feat.extract_stage_2a(file_path).reshape(1, -1)
        p2a = m2a.predict(vec2a, verbose=0)[0]
        final_pred = c2a[np.argmax(p2a)]
    else:
        # --- STAGE 2B: SAD vs RELAXED ---
        vec2b = feat.extract_stage_2b(file_path).reshape(1, -1)
        p2b = m2b.predict(vec2b, verbose=0)[0]
        final_pred = c2b[np.argmax(p2b)]
        
    return final_pred

def evaluate_test_set():
    y_true = []
    y_pred = []
    
    moods = ['angry', 'happy', 'sad', 'relaxed']
    
    print("\n🚀 Memulai Evaluasi Hierarki pada Test Set...")
    
    for mood in moods:
        folder = os.path.join(TEST_DIR, mood)
        if not os.path.exists(folder): continue
        
        files = os.listdir(folder)
        for f in tqdm(files, desc=f"Testing {mood}"):
            if not f.endswith('.wav'): continue
            
            try:
                file_path = os.path.join(folder, f)
                prediction = predict_single(file_path)
                
                y_true.append(mood)
                y_pred.append(prediction)
            except Exception as e:
                print(f"Error {f}: {e}")

    print("\n--- CLASSIFICATION REPORT (HIERARCHICAL) ---")
    print(classification_report(y_true, y_pred))
    
    # Plot CM
    cm = confusion_matrix(y_true, y_pred, labels=moods)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=moods, yticklabels=moods, cmap='Blues')
    plt.title('Confusion Matrix: Experiment 8 (Hierarchy)')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig('cm_exp8_hierarchy.png')
    plt.show()

if __name__ == "__main__":
    evaluate_test_set()