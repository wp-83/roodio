
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
| **Web Application** | [roodio.site](https://roodio.site) |
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

The `machineLearning` module is the brain behind Roodio's mood detection capabilities. It uses a **Hierarchical Multi-Modal Classification** system that branches based on energy level, combining audio signal analysis with NLP-based lyric sentiment analysis.

### 🔥 Pipeline Architecture

The system uses a **3-stage hierarchical pipeline** instead of a single flat classifier:

```
Audio Input
    │
    ▼
┌──────────────────────────────┐
│  Stage 1: Energy Classifier  │  PyTorch Neural Network
│  (YAMNet + RMS + ZCR)        │  → High Energy / Low Energy
└──────────┬───────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ Stage 2A│ │ Stage 2B │
│ (Audio) │ │ (Lyrics) │
│ RF + Meta│ │ BERT     │
│→Angry/  │ │→Sad/     │
│  Happy  │ │  Relaxed │
└─────────┘ └──────────┘
```

1.  **Stage 1 — Energy Classification** (Audio):
    *   **PyTorch Neural Network** (`AudioClassifier`) classifies songs into **High Energy** or **Low Energy**.
    *   Features: **YAMNet** (Transfer Learning) embeddings (mean, std, max) + **RMS** + **ZCR** = 3,074-dimensional vector.
    *   Architecture: `Linear(3074→512) → ReLU → BN → Dropout → Linear(512→256) → ReLU → BN → Dropout → Linear(256→2)`.

2.  **Stage 2A — High Energy Branch** (Audio-only):
    *   **Random Forest + Meta Classifier** (stacking ensemble) to classify between **Angry** and **Happy**.
    *   Uses YAMNet mean embeddings + RMS + ZCR as features.
    *   No lyrics needed — mood distinction is audio-driven.

3.  **Stage 2B — Low Energy Branch** (Lyrics-based):
    *   **Fine-tuned BERT** (`AutoModelForSequenceClassification`) to classify between **Sad** and **Relaxed**.
    *   If no lyrics are provided, defaults to **Relaxed** with 50% confidence.
    *   Lyrics are cleaned and tokenized (max 512 tokens).

### 🧰 ML Libraries & Tools

*   **Core**: `numpy` (<2.0.0), `pandas`, `scipy`
*   **Deep Learning**: `torch` (PyTorch), `tensorflow` (for YAMNet), `transformers` (Hugging Face BERT)
*   **Audio Processing**: `librosa` (RMS, ZCR), `tensorflow_hub` (YAMNet embeddings)
*   **Classical ML**: `scikit-learn` (Random Forest, Meta Classifier), `joblib`
*   **Ops & Tracking**: `mlflow` for experiment tracking
*   **Model Hosting**: Hugging Face Hub (model weights downloaded at runtime)

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
