<?php

use App\Http\Controllers\Admin\OverviewController;
use App\Http\Controllers\Admin\PlaylistController;
use App\Http\Controllers\Admin\SongController;
use App\Http\Controllers\AuthController;
use App\Http\Controllers\SocialController;
use App\Http\Controllers\SuperAdmin\UserController;
use App\Http\Controllers\ThreadController;
use App\Http\Controllers\User\MoodController;
use App\Http\Controllers\User\ProfileController;
use Illuminate\Support\Facades\Route;

Route::get('/welcome', function () {
    return view('errors.503');
})->name('welcome');

// Auth Route
Route::prefix('auth')->group(function () {
    Route::get('/login', [AuthController::class, 'loginView'])->middleware('guest')->name('login');
    Route::post('/login', [AuthController::class, 'login'])->middleware('guest')->name('auth.login');
    Route::post('/logout', [AuthController::class, 'logout'])->name('auth.logout');

    Route::get('email-verification', [AuthController::class, 'emailVerificationView'])->middleware('guest')->name('emailVerification');
    Route::post('email-verification', [AuthController::class, 'emailVerification'])->middleware('guest')->name('auth.emailVerification');

    Route::middleware('forgot.step:otp')->group(function () {
        Route::get('/user-verification', [AuthController::class, 'userVerificationView'])->name('user.verification');
        Route::post('/user-verification', [AuthController::class, 'userVerification'])->name('auth.user.verification');
    });

    Route::middleware('forgot.step:forgot')->group(function () {
        Route::get('/forget-password', [AuthController::class, 'forgetPasswordView'])->name('forgetPassword');
        Route::post('/forget-password', [AuthController::class, 'forgetPassword'])->name('auth.forgetPassword');
    });

    Route::get('/sign-up', [AuthController::class, 'registerView'])->name('register');
    Route::post('/sign-up', [AuthController::class, 'register'])->name('auth.register');

    Route::middleware('register.step:otp')->group(function () {
        Route::get('/otp-authentication', [AuthController::class, 'registerValidationView'])->name('register.validation');
        Route::post('/otp-authentication', [AuthController::class, 'registerValidation'])->name('auth.register.validation');
    });

    Route::middleware('register.step:account')->group(function () {
        Route::get('/create-account', [AuthController::class, 'accountView'])->name('account');
        Route::post('/create-account', [AuthController::class, 'account'])->name('auth.account');
    });
});

// Main Route
Route::prefix('/')->middleware(['auth', 'role:0', 'prevent-back-history'])->group(function () {
    // Profile
    Route::get('profile', [ProfileController::class, 'index'])->name('user.profile');

    // Mood
    Route::post('mood', [MoodController::class, 'moodStore'])->name('mood.store');
    Route::post('preference', [MoodController::class, 'preferenceStore'])->name('preference.store');
    Route::post('mood-update', [MoodController::class, 'moodUpdate'])->name('mood.update');
    Route::post('preference-update', [MoodController::class, 'preferenceUpdate'])->name('preference.update');

    // Threads
    Route::prefix('threads')->group(function () {
        Route::get('', [ThreadController::class, 'index'])->name('thread.index');
        Route::get('/create', [ThreadController::class, 'create'])->name('thread.create');
        Route::post('', [ThreadController::class, 'store'])->name('thread.store');
        Route::post('/{thread}/reply', [ThreadController::class, 'reply'])->name('thread.reply');
    });

    // Social
    Route::prefix('/social')->group(function () {
        Route::get('', [SocialController::class, 'index'])->name('social.index');
    });

    Route::get('', [App\Http\Controllers\User\SongController::class, 'index'])->name('user.index');
    Route::get('{playlists}', [App\Http\Controllers\User\SongController::class, 'playlists'])->name('user.playlists');
});

// Admin Route
Route::prefix('admin')->middleware(['auth', 'role:1', 'prevent-back-history'])->group(function () {
    Route::get('overview', [OverviewController::class, 'overview'])->name('admin.overview');

    Route::prefix('songs')->group(function () {
        Route::get('', [SongController::class, 'index'])->name('admin.songs.index');
        Route::get('/create', [SongController::class, 'create'])->name('admin.songs.create');
        Route::post('/create', [SongController::class, 'store'])->name('admin.songs.store');
        Route::get('/{song}/edit', [SongController::class, 'edit'])->name('admin.songs.edit');
        Route::put('/{song}', [SongController::class, 'update'])->name('admin.songs.update');
        Route::delete('/{song}', [SongController::class, 'destroy'])->name('admin.songs.destroy');
    });

    Route::prefix('playlists')->group(function () {
        Route::get('', [PlaylistController::class, 'index'])->name('admin.playlists.index');
        Route::get('create', [PlaylistController::class, 'create'])->name('admin.playlists.create');
        Route::post('create', [PlaylistController::class, 'store'])->name('admin.playlists.store');
        Route::get('/{playlist}/edit', [PlaylistController::class, 'edit'])->name('admin.playlists.edit');
        Route::put('/{playlist}', [PlaylistController::class, 'update'])->name('admin.playlists.update');
        Route::delete('/{playlist}', [PlaylistController::class, 'destroy'])->name('admin.playlists.destroy');
    });
});

// Super Admin Route
Route::prefix('superadmin')->middleware(['auth', 'role:2', 'prevent-back-history'])->group(function () {
    Route::prefix('users')->group(function () {
        Route::get('', [UserController::class, 'index'])->name('superadmin.users.index');
        Route::get('overview', [UserController::class, 'overview'])->name('superadmin.users.overview');
        Route::get('roles', [UserController::class, 'roles'])->name('superadmin.users.roles');
        Route::post('create', [UserController::class, 'store'])->name('superadmin.users.store');
        Route::put('{user}', [UserController::class, 'update'])->name('superadmin.users.update');
        Route::delete('{user}', [UserController::class, 'destroy'])->name('superadmin.users.destroy');
    });
});

// Thread Route
// Route::post('/{threadId}/reaction', [ThreadController::class, 'react'])->name('thread.react');

// // Mood Route
// Route::get('/moods', [MoodController::class, 'index'])->name('moods.index');
// Route::get('/moods/create', [MoodController::class, 'create'])->name('moods.create');
// Route::post('/moods', [MoodController::class, 'store'])->name('moods.store');
// Route::get('/moods/{mood}/edit', [MoodController::class, 'edit'])->name('moods.edit');
// Route::put('/moods/{mood}', [MoodController::class, 'update'])->name('moods.update');
// Route::delete('/moods/{mood}', [MoodController::class, 'destroy'])->name('moods.destroy');

//unit test route
Route::get('/test/component', function () {
    return view('components.threadBox');
});
