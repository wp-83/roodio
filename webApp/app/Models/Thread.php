<?php
namespace App\Models;

use DB;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Thread extends Model
{
    protected $fillable  = ['userId', 'title', 'content', 'isReplyable', 'datePost'];
    public $incrementing = false;
    protected $keyType   = 'string';

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('threads')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'TH-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'TH-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }

    public function replies(): HasMany
    {
        return $this->hasMany(Reply::class, 'threadId', 'id');
    }

    public function reactions(): HasMany
    {
        return $this->hasMany(Reactions::class, 'threadId', 'id');
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'userId', 'id');
    }
}
