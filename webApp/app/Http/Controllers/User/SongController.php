<?php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\Playlists;

class SongController extends Controller
{
    public function index()
    {
        $playlists    = Playlists::orderByDesc('created_at')->get();
        $username     = auth()->user()->username;
        $fullname     = auth()->user()->userDetail->fullname;
        $profilePhoto = auth()->user()->userDetail->profilePhoo;
        $mood         = 'angry';
        return view('main.index', compact('playlists', 'username', 'fullname', 'profilePhoto', 'mood'));
    }

    public function playlists(Playlists $playlists)
    {
        $songs = $playlists->songs;
        // dd($songs[0]->songPath);
        return view('main.playlists.index', compact('songs'));
    }
}
