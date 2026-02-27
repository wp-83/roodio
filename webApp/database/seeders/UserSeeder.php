<?php

namespace Database\Seeders;

use App\Models\User;
use App\Models\userDetails;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Str;

class UserSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Admin
        // Production: set ADMIN_USERNAME, ADMIN_EMAIL, ADMIN_PASSWORD in .env
        // Local: defaults to admin / admin@gmail.com / password
        $admin = User::create([
            'username' => env('ADMIN_USERNAME', 'admin'),
            'role' => 1,
            'password' => Hash::make(env('ADMIN_PASSWORD', 'password')),
        ]);

        userDetails::create([
            'userId'      => $admin->id,
            'fullname'    => 'Admin User',
            'email'       => env('ADMIN_EMAIL', 'admin@gmail.com'),
            'dateOfBirth' => '1990-01-01',
            'countryId'   => 'ID',
            'gender'      => 1,
        ]);

        // Superadmin
        // Production: set SUPERADMIN_USERNAME, SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD in .env
        // Local: defaults to superadmin / superadmin@gmail.com / password
        $superadmin = User::create([
            'username' => env('SUPERADMIN_USERNAME', 'superadmin'),
            'role' => 2,
            'password' => Hash::make(env('SUPERADMIN_PASSWORD', 'password')),
        ]);

        userDetails::create([
            'userId'      => $superadmin->id,
            'fullname'    => 'Super Admin',
            'email'       => env('SUPERADMIN_EMAIL', 'superadmin@gmail.com'),
            'dateOfBirth' => '1990-01-01',
            'countryId'   => 'ID',
            'gender'      => 1,
        ]);

        // User
        $user = User::create([
            'username' => 'user',
            'role' => 0,
            'password' => Hash::make('password'),
        ]);

        userDetails::create([
            'userId'      => $user->id,
            'fullname'    => 'Regular User',
            'email'       => 'user@gmail.com',
            'dateOfBirth' => '1995-01-01',
            'countryId'   => 'ID',
            'gender'      => 0,
        ]);

        // User Promotion
        $userPromotion = User::create([
            'username' => 'oz123',
            'role' => 0,
            'password' => Hash::make('dummyuser1234567890'),
        ]);

        userDetails::create([
            'userId'      => $userPromotion->id,
            'fullname'    => 'Oliver Brooks',
            'email'       => 'oliver.brooks@gmail.com',
            'dateOfBirth' => '1901-01-01',
            'countryId'   => 'EN',
            'gender'      => 1,
        ]);
    }
}
