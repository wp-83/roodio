<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\DB;

class MoodHistories extends Model
{
    protected $fillable  = ['moodId', 'userId'];
    public $incrementing = false;
    protected $keyType   = 'string';

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('mood_histories')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'MH-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'MH-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
