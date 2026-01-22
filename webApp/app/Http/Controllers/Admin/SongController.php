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
    public function index()
    {
        $songs = Songs::with('user.userDetail')
            ->orderByDesc('created_at')
            ->get();
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
            'song'          => 'required|file|mimes:mp3',
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

        $response = Http::attach(
            'audio_file',
            file_get_contents($song),
            $song->getFilename()
        )->post('https://xullfikar-roodio-analyzer.hf.space/analyze', [
            'lyrics' => $validated['lyrics'],
        ]);

        if ($response->failed()) {
            return back()->withErrors(['api' => 'Failed to send song to API']);
        } else {
            $prediction = $response->json();
        }

        $datas              = $validated;
        $datas['duration']  = $this->getAudioDuration($song->getPathname());
        $datas['songPath']  = $path;
        $datas['photoPath'] = $photoPath;
        $datas['userId']    = Auth::id();
        $datas['moodId']    = $prediction['mood'];
        unset($datas['song']);
        unset($datas['photo']);
        Songs::create($datas);
        return redirect()->route('admin.songs.index')->with('succes', 'Successfully added song');
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

        $validated['userId'] = Auth::id();
        Songs::update($validated);
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
