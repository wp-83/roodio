<?php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\Playlists;

class SongController extends Controller
{
    public function index()
    {
        $playlists = Playlists::with(['songs' => function ($query) {
            $query->applyUserMood();
        }])
            ->whereHas('songs', function ($query) {
                $query->applyUserMood();
            })
            ->orderByDesc('created_at')
            ->get();
        $username     = auth()->user()->username;
        $fullname     = auth()->user()->userDetail->fullname;
        $profilePhoto = auth()->user()->userDetail->profilePhoto;
        $mood         = session('chooseMood', 'happy');
        return view('main.index', compact('playlists', 'username', 'fullname', 'profilePhoto', 'mood'));
    }

    public function playlists(Playlists $playlists)
    {
        $songs = $playlists->songs()->applyUserMood()->get();
        $mood  = session('chooseMood', 'happy');
        return view('main.playlists.index', compact('songs', 'mood'));
    }
}
