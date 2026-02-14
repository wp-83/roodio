<?php
namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;

class SocialController extends Controller
{
    public function index(Request $request)
    {
        $mood = session('chooseMood', 'happy');
        return view('main.socials.index', compact('mood'));
    }
}
