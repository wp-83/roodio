<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Session;

class Songs extends Model
{
    protected $fillable = [
        'userId',
        'moodId',
        'title',
        'lyrics',
        'artist',
        'genre',
        'duration',
        'publisher',
        'datePublished',
        'songPath',
        'photoPath',
    ];

    public $incrementing = false;
    protected $keyType   = 'string';

    public function mood(): BelongsTo
    {
        return $this->belongsTo(Mood::class, 'moodId', 'id');
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'userId', 'id');
    }

    public function playlists()
    {
        return $this->belongsToMany(Playlists::class, 'Tracks', 'songId', 'playlistId');
    }

    public function getOppositesMood($mood)
    {
        $map = [
            'angry'   => 'relaxed',
            'relaxed' => 'angry',
            'happy'   => 'sad',
            'sad'     => 'happy',
        ];

        return $map[$mood] ?? $mood;
    }

    public function scopeApllyUserMood($query)
    {
        $userMood = Session::get('chooseMood');
        $mode     = Session::get('preferenceMood');

        $targetMood = $userMood;

        if ($mode == 'mismatch') {
            $targetMood = $this->getOppositesMood($userMood);
        }

        $map = [
            'happy'   => 'MD-0000001',
            'sad'     => 'MD-0000002',
            'relaxed' => 'MD-0000003',
            'angry'   => 'MD-0000004',
        ];

        return $query->where('moodId', $map[$targetMood]);
    }

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('songs')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'SG-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'SG-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
