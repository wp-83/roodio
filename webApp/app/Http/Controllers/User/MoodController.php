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
        $moodMap = [
            'happy'   => 'MD-0000001',
            'sad'     => 'MD-0000002',
            'relaxed' => 'MD-0000003',
            'angry'   => 'MD-0000004',
        ];

        MoodHistories::create(['moodId' => $moodMap[$session], 'userId' => auth()->user()->id]);
        return redirect()->route('user.index');
    }

    public function preferenceStore(Request $request)
    {
        session()->put('preferenceMood', $request['preference']);
        return redirect()->route('user.index');
    }

    public function moodUpdate(Request $request)
    {
        $validated = $request->validate([
            'mood' => 'required|in:happy,sad,relaxed,angry',
        ]);

        session()->put('chooseMood', $validated['mood']);
        $moodMap = [
            'happy'   => 'MD-0000001',
            'sad'     => 'MD-0000002',
            'relaxed' => 'MD-0000003',
            'angry'   => 'MD-0000004',
        ];

        MoodHistories::where('userId', auth()->id())->latest()->first()->update(['moodId' => $moodMap[$validated['mood']]]);
        return back();
    }

    public function preferenceUpdate(Request $request)
    {
        $validated = $request->validate([
            'preferenceMood' => 'required|in:match,mismatch',
        ]);
        session()->put('preferenceMood', $validated['preferenceMood']);
        return back();
    }
}
