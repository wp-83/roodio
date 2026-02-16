
# 🎵 Roodio - Smart Mood-Based Music Streaming & Analysis Platform

![Laravel](https://img.shields.io/badge/Laravel-12.x-FF2D20?style=for-the-badge&logo=laravel)
![Livewire](https://img.shields.io/badge/Livewire-3.7-4e56a6?style=for-the-badge&logo=livewire)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-4.x-38B2AC?style=for-the-badge&logo=tailwind-css)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow)
![Azure](https://img.shields.io/badge/Azure_Blob_Storage-0078D4?style=for-the-badge&logo=microsoft-azure)

## 📖 Overview

**Roodio** is a cutting-edge music streaming platform that integrates advanced **Machine Learning** to personalize the listening experience based on user mood and emotion. Unlike traditional streaming services, Roodio employs a dual-stack architecture combining a robust **Laravel Web Application** with a sophisticated **Python-based Deep Learning Pipeline** to analyze, classify, and recommend music that resonates with the user's current emotional state.

---

## 🏗️ Architecture

The project is divided into two main modules:

1.  **📱 Web Application (`webApp`)**: A full-stack Laravel application handling the user interface, music streaming, social features, and administrative controls.
2.  **🧠 Machine Learning (`machineLearning`)**: A data science pipeline responsible for audio signal processing, lyric sentiment analysis, and multi-modal mood classification.

---

## 🚀 Web Application

The `webApp` serves as the core platform for users, admins, and super admins. It features a modern, responsive UI built with **TailwindCSS** and **Livewire** for seamless dynamic interactions.

### ✨ Key Features

*   **🎧 Smart Audio Player**:
    *   real-time audio visualizer.
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
*   **☁️ Cloud Infrastructure**:
    *   Song files are securely stored and streamed from **Microsoft Azure Blob Storage**.
    *   MP3 metadata extraction using `james-heinrich/getid3`.

### 🛠️ Tech Stack

*   **Framework**: Laravel 12.x
*   **Frontend**: Livewire 3.7, TailwindCSS 4.x, Alpine.js
*   **Database**: MySQL 8.x
*   **Storage**: Azure Blob Storage
*   **Build Tools**: Vite, PostCSS

---

## 🧠 Machine Learning Engine

The `machineLearning` module is the brain behind Roodio's mood detection capabilities. It utilizes a **Multi-Modal Hybrid Model** that processes both audio signals (spectrograms) and textual data (lyrics) to predict emotional valence and arousal.

### 🔬 Technical Approach

The system employs a multi-stage pipeline:

1.  **Stage 1: Feature Extraction**:
    *   **Audio**: Uses **Librosa** and **YAMNet** (Transfer Learning) to extract deep audio features and Mel-spectrograms from song files.
    *   **Lyrics**: Utilizes **RoBERTa** (Transformer-based NLP) for semantic understanding and sentiment analysis of song lyrics.
2.  **Stage 2: Model Training & Regression**:
    *   **XGBoost Regressor**: Combines extracted features to predict continuous variables for **Valence** (positivity/negativity) and **Arousal** (energy level).
    *   **Deep Mood Aware Augmentation**: Custom data augmentation techniques to balance dataset distribution across emotional quadrants.
3.  **Stage 3: Classification**:
    *   Maps the regression outputs into distinct mood categories (e.g., Happy, Sad, Energetic, Calm).

### 🧰 ML Libraries & Tools

*   **Core**: `numpy` (<2.0.0), `pandas`, `scipy`
*   **Deep Learning**: `tensorflow` (>=2.15), `torch`, `transformers` (HuggingFace)
*   **Audio Processing**: `librosa`, `soundfile`, `audioread`
*   **Classical ML**: `scikit-learn`, `xgboost`
*   **Ops & Tracking**: `mlflow` for experiment tracking.
*   **Data Mining**: `spotipy` (Spotify API) for ground truth labeling.

---

## ⚙️ Installation & Setup

### Prerequisites

*   **PHP** >= 8.2
*   **Composer** (PHP Dependency Manager)
*   **Node.js** & **NPM**
*   **Python** 3.10+ (Ensure compatibility with TensorFlow/NumPy versions)
*   **MySQL** Database

### 1. Web App Setup

```bash
# Navigate to webApp directory
cd webApp

# Install PHP dependencies
composer install

# Install JS dependencies
npm install

# Copy environment file and configure DB/Azure credentials
cp .env.example .env

# Generate App Key
php artisan key:generate

# Run Migrations & Seeders (Initializes Roles, Users, Songs)
php artisan migrate --seed

# Build Frontend Assets
npm run build

# Start Local Server
php artisan serve
```

### 2. Machine Learning Setup

```bash
# Navigate to machineLearning directory
cd machineLearning

# Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# (Optional) Start MLflow UI to view experiments
mlflow ui
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📝 License

This project is open-sourced software licensed under the [MIT license](https://opensource.org/licenses/MIT).

---

<p align="center">
  Built with ❤️ by <strong>Group 3</strong>
</p>