# ── Roodio Laravel Webapp — Dockerfile (Local Dev) ──────────
# Stage 1: Build Node/Vite assets
FROM node:20-alpine AS node-builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --silent

COPY . .
RUN npm run build

# ─────────────────────────────────────────────────────────────
# Stage 2: PHP Runtime (php artisan serve, suitable for local dev)
FROM php:8.2-cli-alpine

WORKDIR /var/www/html

# PHP extensions required by Laravel
RUN apk add --no-cache \
    libpng-dev \
    libjpeg-turbo-dev \
    freetype-dev \
    libzip-dev \
    icu-dev \
    oniguruma-dev \
    curl \
    && docker-php-ext-configure gd --with-freetype --with-jpeg \
    && docker-php-ext-install -j$(nproc) \
        gd \
        pdo_mysql \
        mbstring \
        zip \
        intl \
        bcmath \
        pcntl

# Install Composer
COPY --from=composer:2 /usr/bin/composer /usr/bin/composer

# Copy project files
COPY . .

# Copy built Vite assets from stage 1
COPY --from=node-builder /app/public/build ./public/build

# Install PHP dependencies (without dev deps)
RUN composer install --no-dev --optimize-autoloader --no-interaction --no-scripts

# Storage & cache permissions
RUN mkdir -p storage/framework/{sessions,views,cache} storage/logs bootstrap/cache \
    && chmod -R 775 storage bootstrap/cache

# Copy & set entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
