#!/usr/bin/env bash
# ==============================================================================
#  Roodio - Setup Script (Mac / Linux)
#  Run this in the repo root after git clone:
#    bash setup.sh
# ==============================================================================

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

step()  { echo -e "\n${CYAN}[SETUP] $1${NC}"; }
ok()    { echo -e "${GREEN}[OK] $1${NC}"; }
warn()  { echo -e "${YELLOW}[WARN] $1${NC}"; }
fail()  { echo -e "${RED}[ERROR] $1${NC}"; exit 1; }

echo ""
echo "========================================"
echo -e "${CYAN}  Roodio - Local Setup (Mac/Linux)${NC}"
echo "========================================"

# ==============================================================================
# 1. CHECK PREREQUISITES
# ==============================================================================
step "Checking prerequisites..."

MISSING=()
for cmd in php composer node npm python3 git; do
    command -v "$cmd" >/dev/null 2>&1 || MISSING+=("$cmd")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "\n${RED}  Missing prerequisites:${NC}"
    for m in "${MISSING[@]}"; do echo "    - $m"; done
    echo ""
    echo -e "${YELLOW}  Installation guide (Mac with Homebrew):${NC}"
    echo "    brew install php composer node python git"
    echo "    brew install mysql && brew services start mysql"
    echo ""
    echo -e "${YELLOW}  Installation guide (Linux/Ubuntu):${NC}"
    echo "    sudo apt install php8.2 php8.2-{cli,mbstring,xml,zip,gd,mysql,curl,intl} nodejs npm python3 python3-pip git mysql-server"
    echo "    Composer: https://getcomposer.org/download/"
    fail "Please install the missing prerequisites above, then run this script again."
fi
ok "All prerequisites found."

# ==============================================================================
# 2. SETUP FLASK ML API
# ==============================================================================
step "Setting up Flask ML API..."
cd "$REPO_ROOT/machineLearning/api"

echo "  Creating Python virtual environment..."
python3 -m venv venv
echo "  Activating virtual environment & installing dependencies..."
source venv/bin/activate

python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt -q
ok "Flask ML API dependencies installed in virtual environment."

# ==============================================================================
# 3. SETUP LARAVEL WEBAPP
# ==============================================================================
step "Setting up Laravel Webapp..."
cd "$REPO_ROOT/webApp"

# 3a. Composer install
echo "  Running composer install..."
composer install --no-interaction
ok "Composer dependencies installed."

# 3b. Setup .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    ok ".env created from .env.example"
else
    warn ".env already exists, skipping copy."
fi

# 3c. Point ROODIO_API_URL to local Flask server
if grep -q "ROODIO_API_URL" .env; then
    sed -i.bak "s|^ROODIO_API_URL=.*|ROODIO_API_URL=http://127.0.0.1:7860|" .env && rm -f .env.bak
else
    echo "ROODIO_API_URL=http://127.0.0.1:7860" >> .env
fi
ok "ROODIO_API_URL set to http://127.0.0.1:7860"

# 3d. Generate App Key
php artisan key:generate --quiet
ok "APP_KEY generated."

# 3e. Configure Database
echo ""
echo -e "${YELLOW}  Database Configuration (MySQL):${NC}"
echo "  (Press Enter to use the default value)"
read -p "    DB_HOST     [127.0.0.1]: " DB_HOST;     DB_HOST="${DB_HOST:-127.0.0.1}"
read -p "    DB_PORT     [3306]:      " DB_PORT;     DB_PORT="${DB_PORT:-3306}"
read -p "    DB_DATABASE [roodio]:    " DB_NAME;     DB_NAME="${DB_NAME:-roodio}"
read -p "    DB_USERNAME [root]:      " DB_USER;     DB_USER="${DB_USER:-root}"
read -p "    DB_PASSWORD []:          " DB_PASS

sed -i.bak \
    -e "s|^DB_HOST=.*|DB_HOST=$DB_HOST|" \
    -e "s|^DB_PORT=.*|DB_PORT=$DB_PORT|" \
    -e "s|^DB_DATABASE=.*|DB_DATABASE=$DB_NAME|" \
    -e "s|^DB_USERNAME=.*|DB_USERNAME=$DB_USER|" \
    -e "s|^DB_PASSWORD=.*|DB_PASSWORD=$DB_PASS|" \
    .env && rm -f .env.bak
ok "Database config saved to .env"

# 3f. Run Migrations & Seeders
echo "  Running database migrations and seeders..."
if php artisan migrate --seed --force 2>/dev/null; then
    ok "Database migrated and seeded."
else
    warn "Migration failed. Make sure MySQL is running and credentials in .env are correct."
    warn "Run manually: cd webApp && php artisan migrate --seed"
fi

# 3g. Create storage symlink for local file uploads
php artisan storage:link --quiet 2>/dev/null || true
ok "Storage symlink created (public/storage → storage/app/public)."

# 3h. Install Node dependencies & build assets
echo "  Installing Node.js dependencies..."
npm install
echo "  Building frontend assets (Vite)..."
npm run build
ok "Frontend assets built."

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "========================================"
echo -e "${GREEN}  Setup complete!${NC}"
echo "========================================"
echo ""
echo -e "${CYAN}  To start all servers, run:${NC}"
echo "    bash start.sh"
echo ""
echo -e "${CYAN}  Or manually in 2 separate terminals:${NC}"
echo "    Terminal 1 (ML API) : cd machineLearning/api && python3 app.py"
echo "    Terminal 2 (Webapp) : cd webApp && php artisan serve"
echo ""
echo -e "${CYAN}  Open in browser:${NC}"
echo "    Webapp   http://localhost:8000"
echo "    ML API   http://localhost:7860"
echo ""
