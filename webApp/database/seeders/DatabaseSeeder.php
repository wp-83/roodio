<?php
namespace Database\Seeders;

use App\Models\User;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        // Essential seeders — always run (local + production)
        $this->call(UserSeeder::class);
        $this->call(RegionSeeder::class);
        $this->call(MoodSeeder::class);

        // Production seeders — only run in production
        // These contain pre-loaded songs with Azure storage paths.
        // To run manually: php artisan db:seed --class=SongsSeeder
        if (app()->environment('production')) {
            $this->call(SongsSeeder::class);
            $this->call(PlaylistsSeeder::class);
        }
    }
}
