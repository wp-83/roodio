<?php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\MoodHistories;
use App\Models\Playlists;
use Illuminate\Support\Carbon;
use Illuminate\Support\Facades\Session;

class SongController extends Controller
{
    public function index()
    {
        $user = auth()->user();

        $todaysMood = MoodHistories::where('userId', $user->id)
            ->whereDate('created_at', Carbon::today())
            ->first();

        $moodMapReverse = [
            'MD-0000001' => 'happy',
            'MD-0000002' => 'sad',
            'MD-0000003' => 'relaxed',
            'MD-0000004' => 'angry',
        ];

        if ($todaysMood) {
            if (! session()->has('chooseMood')) {
                $moodName = $moodMapReverse[$todaysMood->moodId] ?? 'happy';

                session()->put('chooseMood', $moodName);

                session()->put('preferenceMood', 'match');
            }
        } else {
            session()->forget('chooseMood');
            session()->forget('preferenceMood');
        }

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
