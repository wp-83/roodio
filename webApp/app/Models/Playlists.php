<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\DB;

class Playlists extends Model
{
    protected $fillable  = ['id', 'userId', 'name', 'description', 'playlistPath'];
    public $incrementing = false;
    protected $keyType   = 'string';

    protected $appends = ['total_duration'];

    public function getTotalDurationAttribute()
    {
        $totalSeconds = $this->songs->sum('duration');
        $hours = floor($totalSeconds / 3600);
        $minutes = floor(($totalSeconds % 3600) / 60);
        
        $duration = [];
        if ($hours > 0) {
            $duration[] = $hours . ' hr';
        }
        if ($minutes > 0 || count($duration) == 0) {
            $duration[] = $minutes . ' min';
        }

        return implode(' ', $duration);
    }

    public function songs()
    {
        return $this->belongsToMany(Songs::class, 'Tracks', 'playlistId', 'songId');
    }

    public function user()
    {
        return $this->belongsTo(User::class, 'userId', 'id');
    }

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('playlists')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'PY-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'PY-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
