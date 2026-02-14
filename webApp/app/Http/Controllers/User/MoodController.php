<?php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\MoodHistories;
use Carbon\Carbon;
use Illuminate\Http\Request;
use App\Models\Mood;
use Illuminate\Support\Facades\DB;

class MoodController extends Controller
{
    public function index()
    {
        $user = auth()->user();
        $moods = MoodHistories::get();

        $startOfWeek = Carbon::now()->startOfWeek();
        $endOfWeek   = Carbon::now()->endOfWeek();
        $today = Carbon::now();

        $endDate = $today->lessThan($endOfWeek) ? $today : $endOfWeek;
        $weekly = auth()->user()
                    ->moodHistories()
                    ->with('mood')
                    ->whereBetween('created_at', [$startOfWeek, $endOfWeek])
                    ->select('moodId', DB::raw('COUNT(*) as total'))
                    ->groupBy('moodId')
                    ->get()
                    ->map(fn($item) => [
                        'id'        => $item->mood->id,
                        'type'      => $item->mood->type,
                        'total'     => $item->total,
                        'startDate' => $startOfWeek->format('F jS, Y'),
                        'endDate'   => $endDate->format('F jS, Y')
                    ]);

        $startOfMonth = Carbon::now()->startOfMonth();
        $endOfMonth   = Carbon::now()->endOfMonth();
        $monthly = auth()->user()
                    ->moodHistories()
                    ->join('moods', 'mood_histories.moodId', '=', 'moods.id')
                    ->whereBetween('mood_histories.created_at', [$startOfMonth, $endOfMonth])
                    ->select(
                        DB::raw('DATE(mood_histories.created_at) as date'),
                        'moods.id as mood_id',
                        'moods.type as mood_type',
                        DB::raw('COUNT(*) as total')
                    )
                    ->groupBy('date', 'moods.id', 'moods.type')
                    ->get()
                    ->groupBy('date')
                    ->map(function ($dayGroup) {
                        $dominantMood = $dayGroup->sortByDesc('total')->first();
                        
                        return [
                            'date'  => $dominantMood->date,
                            'id'    => $dominantMood->mood_id,
                            'type'  => $dominantMood->mood_type,
                            'total' => $dominantMood->total,
                        ];
                    })
                    ->values();
        
        $startOfYear = Carbon::now()->startOfYear();
        $endOfYear   = Carbon::now()->endOfYear();        

        $yearly = auth()->user()
                    ->moodHistories()
                    ->join('moods', 'mood_histories.moodId', '=', 'moods.id')
                    ->whereBetween('mood_histories.created_at', [$startOfYear, $endOfYear])
                    ->select(
                        DB::raw('MONTH(mood_histories.created_at) as month_number'),
                        DB::raw('DATE_FORMAT(mood_histories.created_at, "%M") as month_name'),
                        'moods.type as mood_type',
                        DB::raw('COUNT(*) as total')
                    )
                    ->groupBy('month_number', 'month_name', 'moods.type')
                    ->orderBy('month_number')
                    ->get()
                    ->groupBy('month_number')
                    ->map(function ($monthGroup) use ($startOfYear, $endOfYear) { // TAMBAHKAN USE DI SINI
                        // Ambil mood dominan di bulan ini
                        $dominantMood = $monthGroup->sortByDesc('total')->first();
                        
                        // Hitung total semua mood di bulan ini
                        $totalAll = $monthGroup->sum('total');
                        
                        return [
                            'bulan'      => $dominantMood->month_name,
                            'bulan_ke'   => $dominantMood->month_number,
                            'mood'       => $dominantMood->mood_type,
                            'total'      => $dominantMood->total,
                            'persentase' => round(($dominantMood->total / $totalAll) * 100, 2) . '%',
                            'start_date' => $startOfYear->format('F jS, Y'),
                            'end_date'   => $endOfYear->format('F jS, Y'),
                        ];
                    })
                    ->values();

        $mood = session('chooseMood', 'happy');
        return view('main.moods.index', compact('user', 'weekly', 'monthly', 'yearly', 'mood'));
    }

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
