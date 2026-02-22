import requests
import os

# --- KONFIGURASI ---
# URL Lokal (Sesuai port di app.py)
URL = "http://localhost:7860/predict"

# Nama file lagu yang mau dites (Pastikan file ini ada di folder yang sama!)
FILE_LAGU = r"C:\Users\andiz\Downloads\Set Fire to the Rain.mp3" 

def run_test():
    # 1. Cek apakah file ada
    if not os.path.exists(FILE_LAGU):
        print(f"❌ Error: File '{FILE_LAGU}' tidak ditemukan!")
        print("   -> Tolong copy satu file mp3 ke folder ini dan ubah variabel FILE_LAGU di script.")
        return

    print(f"🚀 Mengirim file '{FILE_LAGU}' ke server...")
    
    try:
        # 2. Buka file dan kirim via POST Request
        with open(FILE_LAGU, 'rb') as f:
            # 'file' adalah key yang diminta oleh app.py (request.files['file'])
            files = {'file': f} 
            
            # Kirim request
            response = requests.post(URL, files=files)
        
        # 3. Cek Respon
        if response.status_code == 200:
            print("\n✅ SUKSES! Server membalas:")
            print("-----------------------------")
            print(response.json())
            print("-----------------------------")
        else:
            print(f"\n⚠️ GAGAL! Kode: {response.status_code}")
            print("Pesan:", response.text)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error Koneksi: Server sepertinya mati.")
        print("   -> Pastikan 'python app.py' sedang jalan di terminal sebelah!")
    except Exception as e:
        print(f"\n❌ Error Lain: {e}")

if __name__ == "__main__":
    run_test()