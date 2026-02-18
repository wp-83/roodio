<?php

namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\ModelFeedback;
use App\Models\ModelLog;
use Illuminate\Http\Request;

class ModelFeedbackController extends Controller
{
    public function store(Request $request)
    {
        $request->validate([
            'song_id' => 'required|exists:songs,id',
            'is_correct' => 'required|boolean',
            'feedback_type' => 'nullable|string|in:implicit,explicit,implicit_skip' // Optional logging context
        ]);

        // Create or Update feedback entry to ensure one unique data point per user per song
        ModelFeedback::updateOrCreate(
            [
                'user_id' => auth()->id(),
                'song_id' => $request->song_id,
            ],
            [
                'is_correct' => $request->is_correct,
                'feedback_type' => $request->feedback_type
            ]
        );

        return response()->json(['message' => 'Feedback logged successfully']);
    }
}
