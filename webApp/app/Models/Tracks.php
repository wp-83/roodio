<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\DB;

class Tracks extends Model
{
    protected $fillable  = ['id', 'songId', 'playlistId'];
    public $incrementing = false;
    protected $keyType   = 'string';

    public function playlists()
    {
        return $this->belongsTo('playlists', 'playlistId');
    }

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('tracks')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'TR-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'TR-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
