<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModelLog extends Model
{
    protected $fillable = [
        'song_id',
        'predicted_mood',
        'confidence_score',
        'is_correct'
    ];

    public function song()
    {
        return $this->belongsTo(Songs::class, 'song_id', 'id');
    }
}
