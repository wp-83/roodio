# 1. Gunakan Python 3.9 (Versi stabil untuk Librosa & Tensorflow)
FROM python:3.9

# 2. Set working directory ke /code
WORKDIR /code

# 3. Install library sistem yang wajib untuk Audio (Librosa)
# libsndfile1 dan ffmpeg wajib ada
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# 4. Copy requirements.txt dulu (supaya caching berjalan)
COPY ./requirements.txt /code/requirements.txt

# 5. Install library Python
# Menggunakan --no-cache-dir agar image lebih kecil
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy seluruh folder proyek ke dalam container
# (Kecuali yang ditulis di .dockerignore)
COPY . /code

# 7. Berikan akses folder cache (untuk Numba/Librosa) agar tidak error permission
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# 8. Jalankan aplikasi
# KARENA main.py ADA DI DALAM FOLDER 'app', GUNAKAN: app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]