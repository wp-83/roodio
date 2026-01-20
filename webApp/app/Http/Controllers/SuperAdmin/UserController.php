<?php
namespace App\Http\Controllers\SuperAdmin;

use App\Http\Controllers\Controller;
use App\Models\User;

class UserController extends Controller
{
    public function index()
    {
        $users        = User::orderByDesc('created_at')->get();
        $totalNewUser = User::whereDate('created_at', today())->count();
        return view('superadmin.index', compact('users', 'totalNewUser'));
    }

    public function overview()
    {
        return view('superadmin.overview');
    }

    public function roles()
    {
        return view('superadmin.roles');
    }
}
