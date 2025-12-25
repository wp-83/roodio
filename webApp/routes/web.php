<?php

use Illuminate\Support\Facades\Route;

// Route::get('/', function () {
//     return view('welcome');
// });

Route::get('/', function () {
    return view('503error');
});

Route::get('/login', function () {
    return view('login');
});

Route::get('/sign-up', function () {
    return view('register');
});

Route::get('/forget-password', function () {
    return view('forgetPass');
});