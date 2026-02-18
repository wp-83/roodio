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
        $admin = User::create([
            'username' => 'admin',
            'role' => 1,
            'password' => Hash::make('password'),
        ]);

        userDetails::create([
            'userId'      => $admin->id,
            'fullname'    => 'Admin User',
            'email'       => 'admin@gmail.com',
            'dateOfBirth' => '1990-01-01',
            'countryId'   => 'ID',
            'gender'      => 1,
        ]);

        // Superadmin
        $superadmin = User::create([
            'username' => 'superadmin',
            'role' => 2,
            'password' => Hash::make('password'),
        ]);

        userDetails::create([
            'userId'      => $superadmin->id,
            'fullname'    => 'Super Admin',
            'email'       => 'superadmin@gmail.com',
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
            'userId'      => $promotion->id,
            'fullname'    => 'Oliver Brooks',
            'email'       => 'oliver.brooks@gmail.com',
            'dateOfBirth' => '1901-01-01',
            'countryId'   => 'EN',
            'gender'      => 1,
        ]);
    }
}
