<?php
namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\Playlists;
use App\Models\Tracks;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;

class PlaylistController extends Controller
{
    public function index(Request $request)
    {
        $query = Playlists::with('user')->withCount('songs');

        if ($request->filled('search')) {
            $search = $request->search;
            $query->where(function ($q) use ($search) {
                $q->where('name', 'like', "%{$search}%")
                    ->orWhere('description', 'like', "%{$search}%")
                    ->orWhereHas('user', function ($u) use ($search) {
                        $u->where('username', 'like', "%{$search}%");
                    });
            });
        }

        if ($request->filled('status')) {
            if ($request->status == 'not_empty') {
                $query->has('songs');
            } elseif ($request->status == 'empty') {
                $query->doesntHave('songs');
            }
        }

        if ($request->filled('sort') && $request->sort == 'oldest') {
            $query->oldest();
        } else {
            $query->latest();
        }

        $playlists = $query->paginate(8);

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
            'image'       => 'required|image|mimes:jpeg,png,jpg|max:5120',
        ]);

        $image = $request->file('image');

        $validated['playlistPath'] = Storage::disk('azure')->put(
            'image',
            $image
        );
        $validated['userId'] = auth()->id();

        $playlist = Playlists::create($validated);

        if ($request->has('songs')) {
            foreach ($request['songs'] as $song) {
                Tracks::create([
                    'songId'     => $song,
                    'playlistId' => $playlist->id,
                ]);
            }
        }

        return redirect()->route('admin.playlists.index')->with(['success' => "Successfully to Create Playlist"]);
    }

    public function edit(Playlists $playlist)
    {
        $currentSongIds = Tracks::where('playlistId', $playlist->id)
            ->pluck('songId')
            ->toArray();

        return view('admin.playlists.edit', compact('playlist', 'currentSongIds'));
    }

    public function update(Request $request, Playlists $playlist)
    {
        $validated = $request->validate([
            'name'        => 'required|max:255',
            'description' => 'required|max:255',
            'image'       => 'nullable|image|max:2048',
        ]);

        if ($request->hasFile('image')) {
            if ($playlist->playlistPath) {
                Storage::disk('azure')->delete($playlist->playlistPath);
            }

            $validated['playlistPath'] = Storage::disk('azure')->put('image', $request->file('image'));
        }

        $playlist->update([
            'name'         => $validated['name'],
            'description'  => $validated['description'],
            'playlistPath' => $validated['playlistPath'] ?? $playlist->playlistPath,
        ]);

        if ($request->has('songs')) {
            Tracks::where('playlistId', $playlist->id)->delete();

            foreach ($request->input('songs') as $songId) {
                Tracks::create([
                    'songId'     => $songId,
                    'playlistId' => $playlist->id,
                ]);
            }
        } else {
            Tracks::where('playlistId', $playlist->id)->delete();
        }

        return redirect()->route('admin.playlists.index')->with('success', 'Playlist updated successfully');
    }

    public function destroy(Playlists $playlist)
    {
        $playlist->delete();
        return redirect()->route('admin.playlists.index')->with('success', 'Playlist deleted successfully');
    }
}
