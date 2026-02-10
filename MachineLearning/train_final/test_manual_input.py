import os
import re
import numpy as np
import librosa
import joblib 
import tensorflow as tf
import tensorflow_hub as hub
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Mute warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

print("🚀 MANUAL SAMPLE TESTING (Global System)")
print("   Audio: Dari folder 'test_samples'")
print("   Lirik: Input Manual di Script")
print("="*60)

# ================= CONFIGURASI =================

# 1. Lokasi Folder Audio Sample
AUDIO_FOLDER = 'data/test_samples' 

# 2. INPUT LIRIK MANUAL DISINI (Format: "nama_file.mp3": "lirik lagu...")
# Pastikan nama file SAMA PERSIS dengan yang ada di folder audio
MANUAL_DATA = {
    "Julien Baker - Appointments (Official Video).mp3": """
        I'm staying in tonight
I won't stop you from leaving
I know that I'm not what you wanted, am I?
Wanted someone who I used to be like
Now you think I'm not trying
Well, don't argue, it's not worth the effort to lie
You don't want to bring it up
And I already know how we look
You don't have to remind me so much
How I disappoint you
It's just that I talked to somebody again
Who knows how to help me get better
Until then I should just try not to miss anymore
Appointments
Ooh ooh ooh ooh
Ooh ooh ooh
I think if I ruin this
That I know I can live with it
Nothing turns out like I pictured it
Maybe the emptiness is just a lesson in canvases
I think if I fail again
That I know you're still listening
Maybe it's all gonna turn out alright
And I know that it's not, but I have to believe that it is
I have to believe that it is
I have to believe that it is
(I have to believe it)
I have to believe that it is
(Probably not, but I have to believe that it is)
And when I tell you that you that it is
Oh, it's not for my benefit
Maybe it's all gonna turn out alright
Oh, I know that it's not, but I have to believe that it is
    """,
    
    "Meghan Trainor - Made You Look (Official Music Video).mp3": """
        I could have my Gucci on
I could wear my Louis Vuitton
But even with nothing on
Bet I made you look (I made you look)
I'll make you double take soon as I walk away
Call up your chiropractor just in case your neck break
Ooh, tell me what you, what you, what you gon' do? Ooh
'Cause I'm 'bout to make a scene, double up that sunscreen
I'm 'bout to turn the heat up, gonna make your glasses steam
Ooh, tell me what you, what you, what you gon' do? Ooh
When I do my walk, walk
I can guarantee your jaw will drop, drop
'Cause they don't make a lot of what I got, got
Ladies, if you feel me, this your bop, bop
(Bop, bop, bop)
I could have my Gucci on (Gucci on)
I could wear my Louis Vuitton
But even with nothing on
Bet I made you look (I made you look)
Yeah, I look good in my Versace dress (take it off)
But I'm hotter when my morning hair's a mess
But even with my hoodie on
Bet I made you look (I made you look)
(Hmm-hmm-hmm)
And once you get a taste, you'll never be the same
This ain't that ordinary, it's that 14 karat cake
Ooh, tell me what you, what you
What you gon' do? Ooh
When I do my walk, walk
I can guarantee your jaw will drop, drop
'Cause they don't make a lot of what I got, got
Ladies, if you feel me, this your bop, bop
(Bop, bop, bop)
I could have my Gucci on (Gucci on)
I could wear my Louis Vuitton
But even with nothing on
Bet I made you look (said I made you look)
Yeah, I look good in my Versace dress (take it off)
But I'm hotter when my morning hair's a mess
But even with my hoodie on
Bet I made you look (said, I made you look)
    """,
    
    "Kesha - Praying (Official Video).mp3": """
        Well, you almost had me fooled
Told me that I was nothing without you
Oh, but after everything you've done
I can thank you for how strong I have become
'Cause you brought the flames and you put me through hell
I had to learn how to fight for myself
And we both know all the truth I could tell
I'll just say this is I wish you farewell
I hope you're somewhere praying, praying
I hope your soul is changing, changing
I hope you find your peace
Falling on your knees, praying
I'm proud of who I am
No more monsters, I can breathe again
And you said that I was done
Well, you were wrong and now the best is yet to come
'Cause I can make it on my own
And I don't need you, I found a strength I've never known
I've been thrown out, I've been burned
When I'm finished, they won't even know your name
You brought the flames and you put me through hell
I had to learn how to fight for myself
And we both know all the truth I could tell
I'll just say this is I wish you farewell
I hope you're somewhere praying, praying
I hope your soul is changing, changing
I hope you find your peace
Falling on your knees, praying
Oh, sometimes, I pray for you at night
Someday, maybe you'll see the light
Oh, some say, in life, you're gonna get what you give
But some things, only God can forgive
I hope you're somewhere praying, praying
I hope your soul is changing, changing
I hope you find your peace
Falling on your knees, praying
    """,
    
    "In The End [Official HD Music Video] - Linkin Park.mp3": """
        [Chester Bennington:]
It starts with one
[Mike Shinoda:]
One thing I don't know why
It doesn't even matter how hard you try
Keep that in mind, I designed this rhyme
To explain in due time

[Chester Bennington:]
All I know
[Mike Shinoda:]
Time is a valuable thing
Watch it fly by as the pendulum swings
Watch it count down to the end of the day
The clock ticks life away

[Chester Bennington:]
It's so unreal
[Mike Shinoda:]
It's so unreal, didn't look out below
Watch the time go right out the window
Trying to hold on, but didn't even know
I wasted it all just to watch you go

[Chester Bennington:]
Watch you go
[Mike Shinoda:]
I kept everything inside and even though I tried, it all fell apart
What it meant to me will eventually be a memory of a time when I tried so hard

[Chester Bennington:]
I tried so hard
And got so far
But in the end
It doesn't even matter
I had to fall
To lose it all
But in the end
It doesn't even matter

[Mike Shinoda:]
One thing, I don't know why
It doesn't even matter how hard you try
Keep that in mind, I designed this rhyme
To remind myself how I tried so hard

[Chester Bennington:]
I tried so hard
[Mike Shinoda:]
In spite of the way you were mocking me
Acting like I was part of your property
Remembering all the times you fought with me
I'm surprised it got so far

[Chester Bennington:]
Got so far
[Mike Shinoda:]
Things aren't the way they were before
You wouldn't even recognize me anymore
Not that you knew me back then
But it all comes back to me in the end

[Chester Bennington:]
In the end
[Mike Shinoda:]
You kept everything inside and even though I tried, it all fell apart
What it meant to me will eventually be a memory of a time when I tried so hard

[Chester Bennington:]
I tried so hard
And got so far
But in the end
It doesn't even matter
I had to fall
To lose it all
But in the end
It doesn't even matter

I've put my trust in you
Pushed as far as I can go
For all this
There's only one thing you should know

I've put my trust in you
Pushed as far as I can go
For all this
There's only one thing you should know

I tried so hard
And got so far
But in the end
It doesn't even matter
I had to fall
To lose it all
But in the end
It doesn't even matter
    """,
    
    # Tambahkan lagu lain di sini...
    "lagu_galau_tapi_santai.mp3": """
        And I hate to say I love you
        When it's so hard for me
        And I hate to say I want you
        When you make it so clear
        You don't want me
    """ 
}

# 3. Path Model
MODEL_DIR = 'models/'
BERT_PATH = 'models/model_hybrid_final' # Path Model BERT Stage 2B

# ================= 1. LOAD MODELS =================
print("⏳ Loading AI Brains...")

try:
    # A. Audio Models
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    s1_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'stage1_nn.h5'))
    s2a_rf = joblib.load(os.path.join(MODEL_DIR, 'stage2a_rf.pkl'))
    s2a_meta = joblib.load(os.path.join(MODEL_DIR, 'stage2a_meta.pkl'))
    
    # B. Text Model (BERT Stage 2B)
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
    bert_model.eval()
    
    # ✅ FIX LABEL MAPPING (Sesuai temuan kamu: 0=sad, 1=relaxed)
    id2label = {0: 'sad', 1: 'relaxed'}
    
    print("✅ System Ready.")

except Exception as e:
    print(f"❌ Error Loading Models: {e}")
    exit()

# ================= 2. HELPER FUNCTIONS =================

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', str(text))
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def trim_middle(y, sr=16000, percentage=0.5):
    if len(y) < sr: return y
    start = int(len(y) * (1 - percentage) / 2)
    end = start + int(len(y) * percentage)
    return y[start:end]

def extract_feat_s1(y, sr):
    y_trim = trim_middle(y, sr)
    if np.max(np.abs(y_trim)) > 0: y_norm = y_trim / np.max(np.abs(y_trim))
    else: y_norm = y_trim
    if len(y_norm) < 16000: y_norm = np.pad(y_norm, (0, 16000 - len(y_norm)))
    _, embeddings, _ = yamnet_model(y_norm)
    yamnet_emb = tf.concat([
        tf.reduce_mean(embeddings, axis=0),
        tf.math.reduce_std(embeddings, axis=0),
        tf.reduce_max(embeddings, axis=0)
    ], axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y_trim))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_trim))
    return np.concatenate([yamnet_emb, [rms, zcr]]).reshape(1, -1)

def extract_feat_s2a_audio(y, sr):
    if len(y) < 16000: y = np.pad(y, (0, 16000-len(y)))
    _, emb, _ = yamnet_model(y)
    vec = tf.reduce_mean(emb, axis=0).numpy()
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.concatenate([vec, [rms, zcr]]).reshape(1, -1)

# ================= 3. TESTING LOGIC =================

def predict_song(filename, lyrics_raw):
    path = os.path.join(AUDIO_FOLDER, filename)
    
    if not os.path.exists(path):
        print(f"❌ File Audio Tidak Ditemukan: {path}")
        return

    print("\n" + "-"*50)
    print(f"🎵 Playing: {filename}")
    print(f"📝 Lyrics : {lyrics_raw[:50]}...") # Print dikit aja

    try:
        # 1. Load Audio
        y, sr = librosa.load(path, sr=16000)
        
        # 2. Stage 1: Energy Check
        feat_s1 = extract_feat_s1(y, sr)
        p1 = s1_model.predict(feat_s1, verbose=0)[0]
        
        # Asumsi: 0=High, 1=Low
        if p1[0] > p1[1]:
            # ---> HIGH ENERGY (Angry/Happy)
            branch = "High Energy"
            f_aud = extract_feat_s2a_audio(y, sr)
            f_txt = np.array([[0.5, 0.5]]) # Dummy text
            
            p_rf = s2a_rf.predict_proba(f_aud)
            meta_in = np.concatenate([p_rf, f_txt], axis=1)
            idx = s2a_meta.predict(meta_in)[0]
            conf = s2a_meta.predict_proba(meta_in)[0][idx]
            
            mood = "ANGRY 😡" if idx == 0 else "HAPPY 😄"
            
        else:
            # ---> LOW ENERGY (Sad/Relaxed) - BERT
            branch = "Low Energy"
            text_clean = clean_text(lyrics_raw)
            
            if len(text_clean) < 5:
                mood = "UNKNOWN (No Lyrics)"
                conf = 0.0
            else:
                inputs = tokenizer(text_clean, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                # probs[0] = sad, probs[1] = relaxed (Sesuai mapping baru)
                
                p_sad = probs[0].item()
                p_rel = probs[1].item()
                
                # --- LOGIKA SENTIMENTAL (AMBIGUITY) ---
                diff = abs(p_sad - p_rel)
                
                if diff < 0.15: # Jika beda tipis (< 15%)
                    mood = "SENTIMENTAL (Sad & Relaxed) 🍂"
                    conf = (p_sad + p_rel) / 2
                elif p_sad > p_rel:
                    mood = "SAD 😢"
                    conf = p_sad
                else:
                    mood = "RELAXED 😌"
                    conf = p_rel

        # Output Result
        bar_len = int(conf * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"📍 Route  : {branch}")
        print(f"🏆 Mood   : {mood}")
        print(f"🎯 Conf   : {conf*100:.1f}% |{bar}|")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ================= 4. RUNNER =================

if __name__ == "__main__":
    if not os.path.exists(AUDIO_FOLDER):
        print(f"⚠️ Folder '{AUDIO_FOLDER}' tidak ditemukan. Buat dulu folder ini dan isi lagu.")
        exit()

    # Loop semua lagu yang ada di Dictionary MANUAL_DATA
    found_any = False
    for fname, lirik in MANUAL_DATA.items():
        predict_song(fname, lirik)
        found_any = True
        
    if not found_any:
        print("⚠️ Dictionary MANUAL_DATA kosong atau tidak ada file yang cocok.")