<?php
namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\Playlists;
use Illuminate\Http\Request;

class PlaylistController extends Controller
{
    public function index()
    {
        $playlists = Playlists::orderByDesc('created_at')->paginate(5);
        return view('admin.playlists.index', compact('playlists'));
    }

    public function create()
    {
        return view('admin.playlists.create');
    }

    public function store(Request $request)
    {
        $validated = $request->validate([
            'name'        => 'required|max:255',
            'description' => 'required|max:255',
        ]);

        Playlists::create($validated);
        return redirect()->route('admin.playlists.index')->with(['success' => "Successfully to Create Playlist"]);
    }

    public function edit(Playlists $playlist)
    {
        return view('admin.playlists.edit', compact('playlist'));
    }

    public function update(Request $request, Playlists $playlist)
    {
        $request->validate([
            'name'        => 'required|max:255',
            'description' => 'required|max:255',
        ]);

        $playlist->update($request->all());
        return redirect()->route('admin.playlists.index')->with('success', 'Playlist updated successfully');
    }

    public function destroy(Playlists $playlist)
    {
        $playlist->delete();
        return redirect()->route('admin.playlists.index')->with('success', 'Playlist deleted successfully');
    }
}
