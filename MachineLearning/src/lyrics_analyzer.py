import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import config as cfg

# --- Download NLTK Resources ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class NRCLexiconAnalyzer:
    def __init__(self, lexicon_path=None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.lexicon = {}
        
        # Daftar kata pembalik makna (Negation words)
        self.negation_words = {
            "not", "no", "never", "don't", "doesn't", "didn't", 
            "won't", "wouldn't", "cant", "cannot", "couldn't", "isn't", "aren't",
            "wasn't", "weren't", "nothing", "neither", "nor"
        }
        
        if lexicon_path is None:
            lexicon_path = os.path.join(cfg.METADATA_DIR, "nrcVadDictionary.txt")
            
        if os.path.exists(lexicon_path):
            self._load_lexicon(lexicon_path)
        else:
            print(f"⚠️ Warning: File kamus tidak ditemukan di {lexicon_path}")

    def _load_lexicon(self, path):
        """Membaca file NRC VAD Dictionary"""
        try:
            df = pd.read_csv(path, sep=r'\t', engine='python', header=0)
            df.columns = df.columns.str.strip().str.lower()
            
            if 'term' not in df.columns:
                df.rename(columns={df.columns[0]: 'term'}, inplace=True)

            # Cek skala negatif
            is_negative_scale = df['valence'].min() < 0

            for _, row in df.iterrows():
                word = str(row['term'])
                v = float(row['valence'])
                a = float(row['arousal'])

                # Normalisasi ke 0..1 jika perlu
                if is_negative_scale:
                    v = (v + 1) / 2
                    a = (a + 1) / 2

                self.lexicon[word] = {'v': v, 'a': a}
        except Exception as e:
            print(f"❌ Gagal membaca file kamus: {e}")

    def preprocess_text(self, text):
        if pd.isna(text) or text == "": return ""
        text = str(text).lower()
        # Simpan tanda petik untuk kata seperti don't, can't
        text = re.sub(r'[^a-z\s\']', '', text) 
        return text

    def predict(self, lyrics_text):
        if not self.lexicon:
            return 0.5, 0.5, {'keywords': []}

        clean_text = self.preprocess_text(lyrics_text)
        words = word_tokenize(clean_text)
        
        val_scores, aro_scores, found_words = [], [], []

        # --- ITERASI DENGAN KONTEKS (Window Size 3) ---
        # Kita melihat kata sebelumnya untuk cek negasi
        for i, word in enumerate(words):
            if word in self.stop_words and word not in self.negation_words: 
                continue
            
            # Cek apakah ini kata emosi?
            score_data = None
            lemma = self.lemmatizer.lemmatize(word)
            
            if lemma in self.lexicon:
                score_data = self.lexicon[lemma]
            elif word in self.lexicon:
                score_data = self.lexicon[word]
            
            if score_data:
                v = score_data['v']
                a = score_data['a']
                
                # --- LOGIKA NEGASI (SMART CHECK) ---
                # Cek 2 kata sebelumnya untuk mencari kata "not", "never", dll.
                # Contoh: "I am NOT happy" atau "I do NOT really like"
                is_negated = False
                if i > 0 and words[i-1] in self.negation_words:
                    is_negated = True
                elif i > 1 and words[i-2] in self.negation_words:
                    is_negated = True
                
                # Jika ada negasi, BALIK nilai Valence
                # (0.9 Jadi 0.1,  0.2 Jadi 0.8)
                if is_negated:
                    v = 1.0 - v
                    # Arousal biasanya sedikit turun jika dinegasikan
                    a = a * 0.8 
                    found_words.append(f"NOT_{word}") # Tandai di log
                else:
                    found_words.append(word)

                val_scores.append(v)
                aro_scores.append(a)

        if len(val_scores) > 0:
            return np.mean(val_scores), np.mean(aro_scores), {
                'keywords': list(set(found_words))[:10]
            }
        else:
            return 0.5, 0.5, {'keywords': []}