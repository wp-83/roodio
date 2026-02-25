#!/bin/sh
set -e

# ── Roodio Webapp Entrypoint ──────────────────────────────────
echo "[entrypoint] Starting Roodio Webapp..."

cd /var/www/html

# 1. Generate APP_KEY if not set
if [ -z "$APP_KEY" ]; then
    echo "[entrypoint] Generating APP_KEY..."
    php artisan key:generate --force
fi

# 2. Clear and cache config
php artisan config:clear
php artisan config:cache

# 3. Wait for MySQL to be ready (retry 30x @ 2 seconds)
echo "[entrypoint] Waiting for MySQL at ${DB_HOST}:${DB_PORT}..."
MAX_TRIES=30
TRIES=0
until php -r "
  try {
    new PDO('mysql:host=${DB_HOST};port=${DB_PORT};dbname=${DB_DATABASE}', '${DB_USERNAME}', '${DB_PASSWORD}');
    exit(0);
  } catch (Exception \$e) { exit(1); }
" 2>/dev/null; do
    TRIES=$((TRIES+1))
    if [ $TRIES -ge $MAX_TRIES ]; then
        echo "[entrypoint] ERROR: MySQL not ready after ${MAX_TRIES} retries."
        exit 1
    fi
    echo "[entrypoint] MySQL not ready yet... ($TRIES/$MAX_TRIES)"
    sleep 2
done
echo "[entrypoint] MySQL is ready!"

# 4. Run migrations
echo "[entrypoint] Running migrations..."
php artisan migrate --force

# 5. Run seeders (idempotent — skips if data already exists)
echo "[entrypoint] Running seeders..."
php artisan db:seed --force 2>/dev/null || echo "[entrypoint] Seeding skipped or already done."

# 6. Create storage symlink
php artisan storage:link 2>/dev/null || true

# 7. Start Laravel development server
echo "[entrypoint] Webapp ready at http://localhost:8000"
exec php artisan serve --host=0.0.0.0 --port=8000
