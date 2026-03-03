# 🎵 Roodio - Smart Mood-Based Music Streaming & Analysis Platform

![Laravel](https://img.shields.io/badge/Laravel-12.x-FF2D20?style=for-the-badge&logo=laravel)
![Livewire](https://img.shields.io/badge/Livewire-3.7-4e56a6?style=for-the-badge&logo=livewire)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-4.x-38B2AC?style=for-the-badge&logo=tailwind-css)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow)
![Azure](https://img.shields.io/badge/Azure_Blob_Storage-0078D4?style=for-the-badge&logo=microsoft-azure)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

## 📖 Table of Contents

1. [Overview](#-overview)
2. [Deployment & Architecture](#-deployment--architecture)
3. [🚀 Web Application Features](#-web-application)
4. [🧠 Machine Learning Engine](#-machine-learning-engine)
5. [💻 Local Setup & Testing (For Reviewers)](#-local-setup--testing)
   - [Prerequisites](#1-prerequisites)
   - [Quick Start](#2-quick-start-automated)
   - [Test Accounts](#4-test-accounts)
   - [Recommended Testing Order](#-recommended-testing-order)
6. [Contributors](#-contributors)

---

## 📖 Overview

**Roodio** is a cutting-edge music streaming platform that integrates advanced **Machine Learning** to personalize the listening experience based on user mood and emotion. Unlike traditional streaming services, Roodio employs a dual-stack architecture combining a robust **Laravel Web Application** with a sophisticated **Python-based Deep Learning Pipeline** to analyze, classify, and recommend music that resonates with the user's current emotional state.

---

## 🏗️ Deployment & Architecture

| Service | URL |
|---|---|
| **Web Application** | [roodio.id](https://roodio.id) |
| **ML API (Hugging Face)** | [Roodio Predict API](https://huggingface.co/spaces/xullfikar/roodio-predict) |

### Architecture Modules

1.  **📱 Web Application (`webApp`)**: A full-stack Laravel application handling the user interface, music streaming, social features, and administrative controls.
2.  **🧠 Machine Learning (`machineLearning`)**: A data science pipeline responsible for audio signal processing, lyric sentiment analysis, and multi-modal mood classification.

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

* **🎧 Smart Audio Player**: Real-time Beat Visualization, interactive vinyl record animation, and full-screen immersive mode with lyrics.
* **😊 Mood Tracking & Analytics**: Daily/weekly mood logging, personalized analytics dashboard, and mood-based playlist generation.
* **👥 Social Community**: Thread discussions, replies, reactions, and user networking.
* **🛡️ Role-Based Access Control**:
    * **User**: Standard streaming and social features.
    * **Admin**: Manage songs (CRUD), playlists, and platform overviews.
    * **Super Admin**: Manage users, roles, and system-wide configurations.
* **🤖 MLOps Dashboard**: Model accuracy monitoring, confidence tracking, and misprediction analysis.

### 🛠️ Tech Stack & Libraries

* **Framework**: Laravel 12.x | Livewire 3.7 | TailwindCSS 4.x | Alpine.js
* **Database & Storage**: MySQL 8.x | Azure Blob Storage
* **UI Components**: Flowbite, ApexCharts, Chart.js, FullCalendar, Tippy.js
* **Animations**: Matter.js (2D Physics), AOS, Canvas Particle Network

---

## 🧠 Machine Learning Engine

The `machineLearning` module uses a **Hierarchical Multi-Modal Classification** system that branches based on energy level, combining audio signal analysis with NLP-based lyric sentiment analysis.

### 🔥 Pipeline Architecture

The system uses a **3-stage hierarchical pipeline**:

```text
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
    * **PyTorch NN** (`AudioClassifier`) classifies songs into **High Energy** or **Low Energy**.
    * Features: **YAMNet** embeddings (mean, std, max) + **RMS** + **ZCR** = 3,074-dimensional vector.
2.  **Stage 2A — High Energy Branch** (Audio-only):
    * **Random Forest + Meta Classifier** (stacking ensemble) to classify between **Angry** and **Happy**.
3.  **Stage 2B — Low Energy Branch** (Lyrics-based):
    * **Fine-tuned BERT** to classify between **Sad** and **Relaxed**.

*Libraries: `torch`, `tensorflow_hub`, `transformers`, `librosa`, `scikit-learn`, `mlflow`*

---

## 💻 Local Setup & Testing

A complete guide to running Roodio on your local machine. Intended for lecturers, reviewers, and developers who want to verify features locally.

### 1. Prerequisites

Make sure the following tools are installed on your machine:

| Tool | Version | Notes (Windows Recommended) |
|---|---|---|
| **PHP + MySQL + Composer** | 8.2+ / 8.x | Install **[Laragon](https://laragon.org/download)** (bundles all three) |
| **Node.js** | 20+ | [Download](https://nodejs.org) |
| **Python** | 3.10+ | [Download](https://python.org/downloads) |
| **Git** | Latest | [Download](https://git-scm.com/downloads) |

### 2. Quick Start (Automated)

> **[!IMPORTANT]**
> **TERMINAL & DIRECTORY REQUIREMENTS:**
> * **Laragon Users:** You **MUST** use the built-in Laragon terminal (Click the **Terminal** button in the Laragon app). Navigate to the `www` folder.
> * **XAMPP Users:** Open your preferred terminal and navigate to the `htdocs` folder.

**Clone the repository:**
```bash
# 1. Navigate to the correct directory first:
# For Laragon:
cd C:\laragon\www
# For XAMPP:
cd C:\xampp\htdocs

# 2. Clone the repository
git clone [https://github.com/Xullfikar/roodio.git](https://github.com/Xullfikar/roodio.git)
cd roodio
```

> **[!WARNING]**
> **CRITICAL:** Before running the setup script, **YOU MUST START YOUR MYSQL SERVER** (e.g., click "Start All" in Laragon or XAMPP). If the database is off, the migration and seeding process will fail.

**Run the Setup Script:**
```cmd
# Windows (CMD / PowerShell / Double-click)
setup.bat

# Mac / Linux
bash setup.sh
```
*The setup script automatically: installs Python/Composer/NPM dependencies, configures `.env`, runs migrations/seeders, creates storage symlinks, and builds frontend assets.*

**Start the Servers:**
```cmd
# Windows (Standard CMD/PowerShell)
start.bat

# Mac / Linux
bash start.sh
```

> **[!NOTE]**
> **FOR LARAGON TERMINAL USERS:**
> If you are using Laragon's built-in terminal, `start.bat` will not work automatically. You must start the servers manually by opening **two separate Laragon terminal tabs/windows**:
>
> **Terminal 1 (Laravel WebApp):**
> ```bash
> cd webApp
> php artisan serve
> ```
>
> **Terminal 2 (Python ML API):**
> ```bash
> cd machineLearning/api
> venv\Scripts\activate
> pip install -r requirements.txt
> python app.py
> ```

If successful, two servers will be running:
* **Laravel Webapp**: `http://localhost:8000`
* **Flask ML API**: `http://localhost:7860`

### 3. Manual Setup (Without Script)

If you prefer to set things up manually, follow these steps in 2 separate terminals (Remember to use Laragon's terminal if you are using Laragon):

**Terminal 1 — ML API (Python)**
```bash
# Ensure you are inside the roodio folder (www/roodio or htdocs/roodio)
cd machineLearning/api

# 1. Create a virtual environment (Recommended)
python -m venv venv     # Windows
python3 -m venv venv    # Mac/Linux

# 2. Activate the virtual environment
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Flask server
python app.py           # Windows
python3 app.py          # Mac/Linux
```

**Terminal 2 — Laravel Webapp**
```bash
# Ensure you are inside the roodio folder (www/roodio or htdocs/roodio)
cd webApp

# 1. Install dependencies
composer install
npm install && npm run build

# 2. Setup Environment
cp .env.example .env
php artisan key:generate

# 3. Setup Database (Ensure MySQL is running first, then configure DB credentials in .env)
php artisan migrate --seed

# 4. Create storage symlink
php artisan storage:link

# 5. Start the server
php artisan serve
```

### 4. Local Environment Details

* **File Uploads**: Stored locally in `webApp/storage/app/public/` (no Azure credentials needed).
* **Emails/OTP**: Emails are NOT sent. OTP codes are logged in `webApp/storage/logs/laravel.log`.
* **Database Initial State**: Starts empty (no pre-loaded songs) to save local storage. Admin must upload songs to test the ML API.

### 5. Test Accounts

After the setup script finishes, use these seeded accounts:

| Role | Username | Password |
|---|---|---|
| **User** | `user` | `password` |
| **Admin** | `admin` | `password` |
| **SuperAdmin**| `superadmin`| `password` |

### 6. Recommended Testing Order

To fully test the application locally, follow this sequence:
1. **Auth**: Test Login, Register (check Laravel log for OTP), and Forgot Password.
2. **Admin Songs**: Login as `admin`, go to Songs, and **upload a few MP3s**. *The Flask ML API will automatically predict the mood.*
3. **Admin Playlists**: Create playlists using the uploaded songs.
4. **User Features**: Login as `user`. Test Home, Mood Filtering, Threads, Socials, and Profile updates.
5. **SuperAdmin**: Login as `superadmin`. Test User Management and Role assignments.
6. **MLOps**: Go back to Admin to check the Model Monitor and Feedback metrics.

---

## 👥 Contributors

* [Andi Zulfikar](https://github.com/Xullfikar) - **Backend Developer & ML Engineer**
* [William Pratama](https://github.com/wp-83) - **Frontend Developer**
* [Agnes Gonxha Febriane Sukma](https://github.com/agnesgonxha) - **UI/UX Designer**
* [Felicia Wijaya](https://github.com/feliciaHw) - **UI/UX Designer**
* [Yoyada Indrayudha](https://github.com/yoyadayudha) - **Quality Assurance**

---
    
## ⚠️ Disclaimer

This project is intended for **educational purposes only**. It is not designed for commercial use, production environments, or widespread deployment. The codebase serves as a demonstration of technical concepts and should be used accordingly.

---

## 📝 License

This project is open-sourced software licensed under the [MIT license](https://opensource.org/licenses/MIT).
