<?php
namespace App\Http\Controllers\SuperAdmin;

use App\Http\Controllers\Controller;
use App\Models\User;
use Illuminate\Http\Request;

class UserController extends Controller
{
    public function index(Request $request)
    {
        $query = User::query();

        if ($request->filled('role')) {
            $query->where('role', $request->get('role'));
        }

        if ($request->filled('search')) {
            $query->where('username', 'LIKE', '%' . $request->get('search') . '%');
        }

        $users        = $query->orderByDesc('created_at')->paginate(10);
        $totalUser    = User::get()->count();
        $totalNewUser = User::whereDate('created_at', today())->count();
        return view('superadmin.index', compact('users', 'totalUser', 'totalNewUser'));
    }

    public function store(Request $request)
    {

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
