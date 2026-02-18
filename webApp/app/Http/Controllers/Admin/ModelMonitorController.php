<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\ModelLog;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Support\Facades\DB;

class ModelMonitorController extends Controller
{
    public function index()
    {
        // 1. Accuracy Trend (Daily) based on FEEDBACK
        $accuracyTrend = \App\Models\ModelFeedback::selectRaw('DATE(created_at) as date, AVG(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) * 100 as accuracy')
            ->groupBy('date')
            ->orderBy('date', 'asc')
            ->take(30)
            ->get();

        // 2. Confidence Score (Prediction Quality)
        // Check if value is 0.0-1.0 or 0-100. Based on screenshots, it's 0-100 (e.g. 77.46)
        $rawAvgConfidence = ModelLog::avg('confidence_score') ?: 0;
        $avgConfidence = $rawAvgConfidence > 1 ? $rawAvgConfidence : $rawAvgConfidence * 100;

        // 3. Mispredicted Songs (Unique candidates for retraining)
        // We group by song_id to avoid "double" rows in the list
        $mispredictedSongs = \App\Models\ModelFeedback::where('is_correct', 0)
            ->select('song_id', DB::raw('count(*) as negative_count'))
            ->groupBy('song_id')
            ->with(['song' => function($q) {
                $q->select('id', 'title', 'artist', 'photoPath');
            }, 'song.modelLog'])
            ->orderByDesc('negative_count')
            ->take(50)
            ->get();

        // 4. Counts
        $totalPredictions = ModelLog::count(); // Total unique songs predicted
        $totalFeedbacks   = \App\Models\ModelFeedback::count();
        $totalCorrect     = \App\Models\ModelFeedback::where('is_correct', 1)->count();
        $totalIncorrect   = \App\Models\ModelFeedback::where('is_correct', 0)->count();
        
        // 5. Overall Accuracy (Based on Feedback)
        $overallAccuracy = $totalFeedbacks > 0 ? ($totalCorrect / $totalFeedbacks) * 100 : 0;

        return view('admin.mlops.index', compact(
            'accuracyTrend', 
            'avgConfidence', 
            'mispredictedSongs',
            'totalPredictions',
            'totalFeedbacks',
            'overallAccuracy'
        ));
    }
}
