<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\DB;

class Playlists extends Model
{
    protected $fillable  = ['id', 'name', 'description', 'playlistPath'];
    public $incrementing = false;
    protected $keyType   = 'string';

    public function tracks()
    {
        return $this->hasMany(Tracks::class, 'playlistId');
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
