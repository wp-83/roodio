<?php
namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;

class SocialController extends Controller
{
    public function index(Request $request)
    {
        $query = User::query()->where('role', '=', '0');

        if ($request->get('filter') === 'following') {
            $users = auth()->user()->followings()->get();
        } else {
            $users = $query->get();
        }

        $mood = session('chooseMood', 'happy');

        return view('main.socials.index', [
            'users' => $users,
            'mood'  => $mood,
        ]);
    }
}
