# 💻 Roodio — Local Setup Guide

A complete guide to running Roodio on your local machine. Intended for developers who want to contribute or testers who want to verify features.

---

## 📋 Prerequisites

Make sure the following tools are installed on your machine:

| Tool | Version | Download |
|---|---|---|
| **PHP** | 8.2+ | Windows: [Laragon](https://laragon.org/download) (includes PHP + MySQL + Composer) |
| **Composer** | Latest | [getcomposer.org](https://getcomposer.org/download/) |
| **Node.js** | 20+ | [nodejs.org](https://nodejs.org) |
| **Python** | 3.10+ | [python.org](https://python.org/downloads) |
| **MySQL** | 8.x | Included in Laragon (Windows) / `brew install mysql` (Mac) |
| **Git** | Latest | [git-scm.com](https://git-scm.com/downloads) |

> **Windows (Recommended):** Install **[Laragon](https://laragon.org/download)** — it bundles PHP, MySQL, and Composer in one package. Install Python and Node.js separately.

---

## 🚀 Quick Start (Automated)

### 1. Clone the Repository

```bash
git clone https://github.com/Xullfikar/roodio.git
cd roodio
```

### 2. Run the Setup Script

**Windows — CMD or double-click:**
```cmd
setup.bat
```

**Windows — PowerShell:**
```powershell
.\setup.ps1
```

**Mac / Linux:**
```bash
bash setup.sh
```

The setup script will automatically:
- ✅ Check all prerequisites
- ✅ Install Python packages for the ML API
- ✅ Install Composer & NPM dependencies
- ✅ Create and configure the `.env` file (interactive DB setup)
- ✅ Run database migrations and seeders
- ✅ Create the storage symlink for local file uploads
- ✅ Build Vite/Tailwind frontend assets

### 3. Start the Servers

**Windows — CMD or double-click:**
```cmd
start.bat
```

**Windows — PowerShell:**
```powershell
.\start.ps1
```

**Mac / Linux:**
```bash
bash start.sh
```

Two servers will start:

| Server | URL | Description |
|---|---|---|
| Laravel Webapp | http://localhost:8000 | Main application |
| Flask ML API | http://localhost:7860 | AI mood detection |

---

## 🔧 Manual Setup (Without Script)

If you prefer to set things up manually, run the following in 2 separate terminals:

```bash
# Terminal 1 — ML API
cd machineLearning/api
pip install -r requirements.txt
python app.py       # Windows
python3 app.py      # Mac/Linux

# Terminal 2 — Laravel
cd webApp
composer install
cp .env.example .env
php artisan key:generate
php artisan migrate --seed
php artisan storage:link
npm install && npm run build
php artisan serve
```

---

## 🔐 Test Accounts

After seeding, 3 test accounts are available:

| Role | Username | Email | Password |
|---|---|---|---|
| User | `user` | `user@gmail.com` | `password` |
| Admin | `admin` | `admin@gmail.com` | `password` |
| SuperAdmin | `superadmin` | `superadmin@gmail.com` | `password` |

---

## ⚙️ Environment Configuration

The `.env` file is automatically configured by the setup script. Here are the important settings for local development:

| Variable | Value | Description |
|---|---|---|
| `APP_ENV` | `local` | Environment mode |
| `APP_URL` | `http://localhost:8000` | Application base URL |
| `FILESYSTEM_DISK` | `public` | Uploaded files are stored in local storage |
| `MAIL_MAILER` | `log` | Emails are not sent — check `storage/logs/laravel.log` |
| `ROODIO_API_URL` | `http://127.0.0.1:7860` | Flask ML API endpoint |

### 📧 OTP / Email Verification

Since `MAIL_MAILER=log`, emails are **not actually sent** to any inbox. To view OTP codes:

```bash
# Open the Laravel log file
cat webApp/storage/logs/laravel.log
# Look for the most recent 6-digit code
```

### 📁 File Uploads

Uploaded files (profile photos, song covers, MP3s) are stored in `webApp/storage/app/public/` and served via `http://localhost:8000/storage/`. The storage symlink is created automatically by the setup script.

---

## 📦 Database Seeding

In local development, only **essential seeders** run automatically:

| Seeder | Contents | Runs Locally? |
|---|---|---|
| **UserSeeder** | 3 test accounts (user, admin, superadmin) | ✅ Yes |
| **RegionSeeder** | Country list | ✅ Yes |
| **MoodSeeder** | 4 moods (happy, sad, relaxed, angry) | ✅ Yes |
| **SongsSeeder** | 179 pre-loaded songs | ❌ No (production only) |
| **PlaylistsSeeder** | Pre-loaded playlists | ❌ No (production only) |

> **The database starts empty** (no songs or playlists). Log in as Admin and upload songs via the dashboard to start testing.

To manually run the production seeders if needed:
```bash
cd webApp
php artisan db:seed --class=SongsSeeder
php artisan db:seed --class=PlaylistsSeeder
```

---

## 🐳 Docker (Alternative)

To run the entire stack using Docker:

```bash
docker-compose up --build
```

Docker will spin up 3 services: MySQL, Flask ML API, and the Laravel Webapp.

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| Migration failed | Make sure MySQL is running and credentials in `.env` are correct |
| ML API error | Make sure Python 3.10+ is installed and `pip install -r requirements.txt` succeeded |
| Images not showing | Run `php artisan storage:link` inside the `webApp` folder |
| OTP not appearing | Check `webApp/storage/logs/laravel.log` |
| Port 8000 already in use | Kill the other process or use `php artisan serve --port=8001` |
| `npm run build` error | Make sure Node.js 20+ is installed |

---

## 📌 Recommended Testing Order

1. **Auth** — Login, Register, Forgot Password
2. **Admin Songs** — Upload a few songs (requires Flask ML API to be running)
3. **Admin Playlists** — Create playlists from the uploaded songs
4. **User Features** — Home, Moods, Threads, Socials, Profile
5. **SuperAdmin** — User management, Roles
6. **MLOps** — Model monitor, Feedback

---

> Back to [main README](README.md)
