<?php
namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\Playlists;
use App\Models\Songs;

class OverviewController extends Controller
{
    public function overview()
    {
        $totalSongs     = Songs::count();
        $totalPlaylists = Playlists::count();

        // 2. Data Terbaru (Limit 5)
        $recentSongs     = Songs::latest()->take(5)->get();
        $recentPlaylists = Playlists::with('user')->latest()->take(5)->get();

        return view('admin.overview', compact(
            'totalSongs',
            'totalPlaylists',
            'recentSongs',
            'recentPlaylists'
        ));
    }
}
