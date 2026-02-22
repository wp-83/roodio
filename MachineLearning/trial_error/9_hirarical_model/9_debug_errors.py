import os
import pandas as pd
import features_hierarchy as feat
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Load Models
MODELS_DIR = 'models_hierarchy'
m1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_1_arousal.h5'))
c1 = np.load(os.path.join(MODELS_DIR, 'classes_stage_1_arousal.npy'), allow_pickle=True)
m2a = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2a_high_group.h5'))
c2a = np.load(os.path.join(MODELS_DIR, 'classes_stage_2a_high_group.npy'), allow_pickle=True)
m2b = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_stage_2b_low_group.h5'))
c2b = np.load(os.path.join(MODELS_DIR, 'classes_stage_2b_low_group.npy'), allow_pickle=True)

def predict_debug(file_path):
    # Stage 1
    vec1 = feat.extract_stage_1(file_path).reshape(1, -1)
    p1 = m1.predict(vec1, verbose=0)[0]
    s1_pred = c1[np.argmax(p1)]
    
    final = ""
    if s1_pred == 'high':
        vec2 = feat.extract_stage_2a(file_path).reshape(1, -1)
        p2 = m2a.predict(vec2, verbose=0)[0]
        final = c2a[np.argmax(p2)]
    else:
        vec2 = feat.extract_stage_2b(file_path).reshape(1, -1)
        p2 = m2b.predict(vec2, verbose=0)[0]
        final = c2b[np.argmax(p2)]
    return final

# Scan Test Folder
TEST_DIR = 'data/processed_exp4/test'
errors = []

print("🕵️‍♂️ Menganalisis Kesalahan...")
for true_label in ['angry', 'happy', 'sad', 'relaxed']:
    folder = os.path.join(TEST_DIR, true_label)
    if not os.path.exists(folder): continue
    
    for f in os.listdir(folder):
        if not f.endswith('.wav'): continue
        pred = predict_debug(os.path.join(folder, f))
        
        if pred != true_label:
            errors.append({
                'File': f,
                'True': true_label,
                'Predicted': pred
            })

# Show Report
df = pd.DataFrame(errors)
print("\n=== DAFTAR TERSANGKA KESALAHAN ===")
if not df.empty:
    print(df.to_string())
    df.to_csv('error_report.csv', index=False)
else:
    print("Hebat! Tidak ada error (Mustahil).")