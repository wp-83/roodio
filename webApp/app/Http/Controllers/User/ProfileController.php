<?php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;

class ProfileController extends Controller
{
    public function index()
    {
        dd('masuk');
        return view('main.profile');
    }
}
