<?php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\MoodHistories;
use Illuminate\Http\Request;

class MoodController extends Controller
{
    public function moodStore(Request $request)
    {
        session()->put('chooseMood', $request['mood']);
        $session = session('chooseMood');
        if ($session === 'happy') {
            $mood = 'MD-0000001';
        } else if ($session === 'sad') {
            $mood = 'MD-0000002';
        } else if ($session === 'relaxed') {
            $mood = 'MD-0000003';
        } else if ($session === 'angry') {
            $mood = 'MD-0000004';
        }

        MoodHistories::create(['moodId' => $mood, 'userId' => auth()->user()->id]);
        return redirect()->route('user.index');
    }

    public function preferenceStore(Request $request)
    {
        session()->put('preferenceMood', $request['preference']);
        return redirect()->route('user.index');
    }
}
