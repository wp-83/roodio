<?php
namespace App\Models;

// use Illuminate\Contracts\Auth\MustVerifyEmail;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Relations\BelongsToMany;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\HasOne;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Illuminate\Notifications\Notifiable;
use Illuminate\Support\Facades\DB;

class User extends Authenticatable
{
    /** @use HasFactory<\Database\Factories\UserFactory> */
    use HasFactory, Notifiable;

    /**
     * The attributes that are mass assignable.
     *
     * @var list<string>
     */
    protected $fillable = [
        'username',
        'password',
        'role',
    ];

    protected $primaryKey = 'id';

    public $incrementing = false;
    protected $keyType   = 'string';

    /**
     * The attributes that should be hidden for serialization.
     *
     * @var list<string>
     */
    protected $hidden = [
        'password',
        'remember_token',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'email_verified_at' => 'datetime',
            'password'          => 'hashed',
        ];
    }

    public function songs(): HasMany
    {
        return $this->hasMany(Songs::class, 'userId', 'id');
    }

    public function userDetail(): HasOne
    {
        return $this->hasOne(userDetails::class, 'userId', 'id');
    }

    public function moodHistories()
    {
        return $this->hasMany(MoodHistories::class, 'userId');
    }

    public function moods()
    {
        return $this->belongsToMany(Mood::class, 'mood_histories', 'userId', 'moodId');
    }

    public function followings(): BelongsToMany
    {
        return $this->belongsToMany(User::class, 'follows', 'userId', 'followedId');
    }

    public function followers(): BelongsToMany
    {
        return $this->belongsToMany(User::class, 'follows', 'followedId', 'userId');
    }

    /**
     * Scope to search users by username or fullname
     */
    public function scopeSearchUsers($query, $search)
    {
        if (!$search) {
            return $query;
        }

        return $query->where(function ($q) use ($search) {
            $q->where('username', 'LIKE', "%{$search}%")
              ->orWhereHas('userDetail', function ($q) use ($search) {
                  $q->where('fullname', 'LIKE', "%{$search}%");
              });
        });
    }

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('users')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'US-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'US-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
