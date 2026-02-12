<?php
namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\Songs;
use getID3;
use Http;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;

class SongController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index(Request $request)
    {
        $query = Songs::query()->with('user.userDetail');

        if ($request->filled('search')) {
            $search = $request->search;
            $query->where(function ($q) use ($search) {
                $q->where('title', 'like', "%{$search}%")
                    ->orWhere('artist', 'like', "%{$search}%")
                    ->orWhere('publisher', 'like', "%{$search}%");
            });
        }

        if ($request->filled('mood')) {
            $query->whereHas('mood', function ($q) use ($request) {
                $q->where('type', $request->mood);
            });
        }

        $songs = $query->latest()->paginate(perPage: 10);

        return view('admin.songs.index', compact('songs'));
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        return view('admin.songs.create');
    }

    protected function getAudioDuration($filePath)
    {
        $getID3          = new getID3();
        $fileInfo        = $getID3->analyze($filePath);
        $durationSeconds = $fileInfo['playtime_seconds'] ?? 0;
        return $durationSeconds;
    }

    /**
     * Store a newly created resource in storage.
     */

    public function store(Request $request)
    {
        $validated = $request->validate([
            'title'         => 'required|max:255',
            'lyrics'        => 'required',
            'artist'        => 'required|max:255',
            'genre'         => 'required|max:255',
            'publisher'     => 'required|max:255',
            'datePublished' => 'required|date',
            'song'          => 'required|file|mimes:mp3|max:10240',
            'photo'         => 'required|image|mimes:jpeg,png,jpg|max:5120',
        ]);

        $song  = $request->file('song');
        $photo = $request->file('photo');

        $path = Storage::disk('azure')->put(
            'songs',
            $song
        );

        $photoPath = Storage::disk('azure')->put(
            'images',
            $photo
        );

        try {
            $apiUrl = env('ROODIO_API_URL') . '/predict';

            $response = Http::attach(
                'file',
                file_get_contents($song->getRealPath()),
                $song->getClientOriginalName()
            )->post($apiUrl, [
                'lyrics' => $validated['lyrics'],
            ]);

            if ($response->successful()) {
                $result         = $response->json();
                $moodId         = $result['data']['mood_id'] ?? null;
                $moodConfidence = $result['data']['confidence'] ?? null;
            } else {
                return back()->withErrors(['api' => 'AI Server Error: ' . $response->body()]);
            }

        } catch (\Exception $e) {
            return back()->withErrors(['api' => 'Connection Failed: ' . $e->getMessage()]);
        }

        $datas              = $validated;
        $datas['duration']  = $this->getAudioDuration($song->getPathname());
        $datas['songPath']  = $path;
        $datas['photoPath'] = $photoPath;
        $datas['userId']    = Auth::id();

        $datas['moodId']     = $moodId;
        $datas['confidence'] = $moodConfidence;

        unset($datas['song']);
        unset($datas['photo']);

        Songs::create($datas);

        return redirect()->route('admin.songs.index')->with('success', 'Successfully added song with AI prediction!');
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(Songs $song)
    {
        return view('admin.songs.edit', compact('song'));
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, Songs $song)
    {
        $validated = $request->validate([
            'title'         => 'required|max:255',
            'lyrics'        => 'required',
            'artist'        => 'required|max:255',
            'genre'         => 'required|max:255',
            'publisher'     => 'required|max:255',
            'datePublished' => 'required|date',
        ]);

        if ($request->hasFile('photo')) {
            $photo     = $request->file('photo');
            $photoPath = Storage::disk('azure')->put(
                'images',
                $photo
            );
            $song->update(['photoPath' => $photoPath]);
        }

        $validated['userId'] = Auth::id();
        $song->update($validated);
        return redirect()->route('admin.songs.index')->with('success', 'Song updated successfully');
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(Songs $song)
    {
        $song->delete();
        return redirect()->route('admin.songs.index')->with('success', 'Song deleted successfully');
    }
}
