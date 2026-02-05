<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Facades\DB;

class Follow extends Model
{
    protected $fillable = ['userId', 'followedId'];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'userId', 'id');
    }

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('follows')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'FW-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'FW-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
