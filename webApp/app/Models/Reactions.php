<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Reactions extends Model
{
    protected $fillable  = ['userId', 'threadId'];
    public $incrementing = false;
    protected $keyType   = 'string';
}
