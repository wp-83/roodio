<?php
namespace App\Models;

use DB;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasOne;

class Mood extends Model
{
    protected $fillable  = ['type'];
    public $incrementing = false;
    protected $keyType   = 'string';

    public function songs(): HasOne
    {
        return $this->hasOne(Songs::class);
    }

    public function users()
    {
        return $this->belongsToMany(User::class, 'mood_histories', 'moodId', 'userId');
    }

    public function moods()
    {
        return $this->hasMany(MoodHistories::class, 'moodId');
    }

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('moods')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'MD-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'MD-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
