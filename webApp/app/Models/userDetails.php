<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Facades\DB;

class userDetails extends Model
{
    protected $fillable  = ['fullname', 'email', 'dateOfBirth', 'countryId', 'gender', 'userId'];
    public $incrementing = false;
    protected $keyType   = 'string';

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'userId', 'id');
    }

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('user_details')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'UD-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'UD-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
