<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModelFeedback extends Model
{
    protected $fillable = ['song_id', 'user_id', 'is_correct', 'feedback_type'];
    public $timestamps = true;

    // Use default table name 'model_feedbacks' or specify if needed
    protected $table = 'model_feedbacks';

    public function song()
    {
        return $this->belongsTo(Songs::class, 'song_id', 'id');
    }
}
