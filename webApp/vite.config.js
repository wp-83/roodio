import { defineConfig } from 'vite';
import laravel from 'laravel-vite-plugin';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
    plugins: [
        laravel({
            input: ['resources/css/app.css', 'resources/js/app.js'],
            refresh: true,
        }),
        tailwindcss(),
    ],
    server: {
        // host: '0.0.0.0', // Mengizinkan akses dari semua IP
        // hmr: {
        //     host: '10.68.99.122' // GANTI dengan IP Address komputer kamu dari langkah 1, php artisan serve --host=0.0.0.0 --port=8000
        // },
        watch: {
            ignored: ['**/storage/framework/views/**'],
        },
    },
});
