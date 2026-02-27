
# 🎵 Roodio - Smart Mood-Based Music Streaming & Analysis Platform

![Laravel](https://img.shields.io/badge/Laravel-12.x-FF2D20?style=for-the-badge&logo=laravel)
![Livewire](https://img.shields.io/badge/Livewire-3.7-4e56a6?style=for-the-badge&logo=livewire)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-4.x-38B2AC?style=for-the-badge&logo=tailwind-css)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow)
![Azure](https://img.shields.io/badge/Azure_Blob_Storage-0078D4?style=for-the-badge&logo=microsoft-azure)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

## 📖 Overview

**Roodio** is a cutting-edge music streaming platform that integrates advanced **Machine Learning** to personalize the listening experience based on user mood and emotion. Unlike traditional streaming services, Roodio employs a dual-stack architecture combining a robust **Laravel Web Application** with a sophisticated **Python-based Deep Learning Pipeline** to analyze, classify, and recommend music that resonates with the user's current emotional state.

---

## 🏗️ Architecture

The project is divided into two main modules:

1.  **📱 Web Application (`webApp`)**: A full-stack Laravel application handling the user interface, music streaming, social features, and administrative controls.
2.  **🧠 Machine Learning (`machineLearning`)**: A data science pipeline responsible for audio signal processing, lyric sentiment analysis, and multi-modal mood classification.

---

## 🌐 Deployment

| Service | URL |
|---|---|
| **Web Application** | [roodio.id](https://roodio.id) |
| **ML API (Hugging Face)** | [Roodio Predict API](https://huggingface.co/spaces/xullfikar/roodio-predict) |

### Production Infrastructure

| Component | Technology |
|---|---|
| Cloud Server (VM) | Azure VM |
| File Storage | Azure Blob Storage |
| Database | MySQL 8.x |
| Email Service | Brevo (SMTP) |
| CDN & Security | Cloudflare |
| Domain | Hostinger |

---

## 🚀 Web Application

The `webApp` serves as the core platform for users, admins, and super admins. It features a modern, responsive UI built with **TailwindCSS** and **Livewire** for seamless dynamic interactions.

### ✨ Key Features

*   **🎧 Smart Audio Player**:
    *   **Real-time Beat Visualization** & Audio Spectrum.
    *   Interactive vinyl record animation.
    *   Full-screen immersive mode with lyrics and queue management.
*   **😊 Mood Tracking & Analytics**:
    *   Daily, weekly, and yearly mood logging.
    *   Personalized analytics dashboard.
    *   Mood-based playlist generation.
*   **👥 Social Community**:
    *   **Threads**: Users can create discussions, reply to others, and react to posts.
    *   **Socials**: Discover and connect with other users.
*   **🛡️ Role-Based Access Control (RBAC)**:
    *   **User**: Standard streaming and social features.
    *   **Admin**: Manage songs (CRUD), playlists, and view platform overviews.
    *   **Super Admin**: Manage users, roles, and system-wide configurations.
*   **🤖 MLOps Dashboard**:
    *   Model accuracy monitoring and confidence tracking.
    *   User feedback loop for continuous improvement.
    *   Misprediction analysis.

### 🛠️ Tech Stack

*   **Framework**: Laravel 12.x
*   **Frontend**: Livewire 3.7, TailwindCSS 4.x, Alpine.js
*   **Database**: MySQL 8.x
*   **Storage**: Azure Blob Storage
*   **Build Tools**: Vite, PostCSS

### 🎨 Frontend Libraries & Tools

*   **UI Components**: [Flowbite](https://flowbite.com/)
*   **Charts & Visualizations**: [ApexCharts](https://apexcharts.com/), [Chart.js](https://www.chartjs.org/)
*   **Calendar**: [FullCalendar](https://fullcalendar.io/)
*   **Tooltips & Popups**: [Popper.js](https://popper.js.org/), [Tippy.js](https://atomiks.github.io/tippyjs/)
*   **Animations & Physics**:
    *   [Matter.js](https://brm.io/matter-js/) (2D Physics Engine)
    *   [AOS](https://michalsnik.github.io/aos/) (Animate On Scroll)
    *   [Canvas Particle Network](https://github.com/JulianLaval/canvas-particle-network)

---

## 🧠 Machine Learning Engine

The `machineLearning` module is the brain behind Roodio's mood detection capabilities. It utilizes a **Multi-Modal Hybrid Model** that processes both audio signals (spectrograms) and textual data (lyrics) to predict emotional valence and arousal.

### 🔥 Core Module (`machineLearning/finished`)

The production-ready models and training scripts are located in the `machineLearning/finished` directory:

*   **`train_stage1_pytorch.py`**: Stage 1 — Audio feature extraction and classification using PyTorch.
*   **`train_stage2a_angry_happy.py`**: Stage 2A — Refinement for specific mood quadrants (Angry/Happy).
*   **`lyrics_stage2b.ipynb`**: Stage 2B — NLP pipeline and sentiment analysis using RoBERTa.
*   **`test_manual_input.py`**: Utility for manual model testing.

### 🔬 Technical Approach

The system employs a multi-stage pipeline:

1.  **Stage 1 — Feature Extraction**:
    *   **Audio**: Uses **Librosa** and **YAMNet** (Transfer Learning) to extract deep audio features and Mel-spectrograms.
    *   **Lyrics**: Utilizes **RoBERTa** (Transformer-based NLP) for semantic understanding and sentiment analysis.
2.  **Stage 2 — Model Training & Regression**:
    *   **XGBoost Regressor**: Combines extracted features to predict continuous variables for **Valence** and **Arousal**.
    *   **Deep Mood Aware Augmentation**: Custom data augmentation to balance dataset distribution across emotional quadrants.
3.  **Stage 3 — Classification**:
    *   Maps the regression outputs into distinct mood categories (Happy, Sad, Relaxed, Angry).

### 🧰 ML Libraries & Tools

*   **Core**: `numpy` (<2.0.0), `pandas`, `scipy`
*   **Deep Learning**: `tensorflow` (>=2.15), `torch` (PyTorch), `transformers` (Hugging Face)
*   **Audio Processing**: `librosa`, `soundfile`, `audioread`
*   **Classical ML**: `scikit-learn`, `xgboost`
*   **Ops & Tracking**: `mlflow` for experiment tracking.
*   **Data Mining**: `spotipy` (Spotify API) for ground truth labeling.

---

## 💻 Local Development

Want to run Roodio on your local machine? See the full setup guide:

### 👉 [LOCAL_SETUP.md](LOCAL_SETUP.md)

Quick start:
```bash
git clone https://github.com/Xullfikar/roodio.git
cd roodio
setup.bat        # Windows
bash setup.sh    # Mac/Linux
```

---

## 👥 Contributors

*   [Andi Zulfikar](https://github.com/Xullfikar) - **Backend Developer**
*   [William Pratama](https://github.com/wp-83) - **Frontend Developer**
*   [Agnes Gonxha Febriane Sukma](https://github.com/agnesgonxha) - **UI/UX Designer**
*   [Felicia Wijaya](https://github.com/feliciaHw) - **UI/UX Designer**
*   [Yoyada Indrayudha](https://github.com/yoyadayudha) - **Quality Assurance**

---
    
## ⚠️ Disclaimer

This project is intended for **educational purposes only**. It is not designed for commercial use, production environments, or widespread deployment. The codebase serves as a demonstration of technical concepts and should be used accordingly.

---

## 📝 License

This project is open-sourced software licensed under the [MIT license](https://opensource.org/licenses/MIT).
