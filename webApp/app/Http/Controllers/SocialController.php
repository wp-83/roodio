<?php
namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;

class SocialController extends Controller
{
    public function index(Request $request)
    {
        $search = $request->get('search');
        $filter = $request->get('filter');
        
        if ($filter === 'following') {
            $users = auth()->user()->followings();
            
            // Apply search to following users
            if ($search) {
                $users = $users->where(function ($q) use ($search) {
                    $q->where('username', 'LIKE', "%{$search}%")
                      ->orWhereHas('userDetail', function ($q) use ($search) {
                          $q->where('fullname', 'LIKE', "%{$search}%");
                      });
                });
            }
            
            $users = $users->get();
        } else {
            $query = User::query()->where('role', '=', '0');
            $query->searchUsers($search);
            $users = $query->get();
        }

        $mood = session('chooseMood', 'happy');

        return view('main.socials.index', [
            'users'  => $users,
            'mood'   => $mood,
            'search' => $search,
        ]);
    }
}
