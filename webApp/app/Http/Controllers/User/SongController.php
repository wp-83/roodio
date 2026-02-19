<?php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\MoodHistories;
use App\Models\Playlists;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Support\Facades\Session;

class SongController extends Controller
{
    public function index(Request $request)
    {
        $user = auth()->user();
        $search = $request->get('search');

        $todaysMood = MoodHistories::where('userId', $user->id)
            ->whereDate('created_at', Carbon::today())
            ->first();

        $moodMapReverse = [
            'MD-0000001' => 'happy',
            'MD-0000002' => 'sad',
            'MD-0000003' => 'relaxed',
            'MD-0000004' => 'angry',
        ];

        if ($todaysMood) {
            if (! session()->has('chooseMood')) {
                $moodName = $moodMapReverse[$todaysMood->moodId] ?? 'happy';

                session()->put('chooseMood', $moodName);

                session()->put('preferenceMood', 'match');
            }
        } else {
            session()->forget('chooseMood');
            session()->forget('preferenceMood');
        }

        $mood = session('chooseMood', 'happy');

        // Base Query with Mood Filtering
        $queryBuilder = Playlists::with(['songs' => function ($query) use ($search) {
            $query->applyUserMood();
            if ($search) {
                $query->searchSongs($search);
            }
        }])
        ->whereHas('songs', function ($query) use ($search) {
            $query->applyUserMood();
            if ($search) {
                $query->searchSongs($search);
            }
        });

        // 1. New Arrivals (Priority: Latest created)
        $newArrivalPlaylists = (clone $queryBuilder)->orderByDesc('created_at')->take(6)->get();

        // Collect IDs to exclude
        $excludeIds = $newArrivalPlaylists->pluck('id')->toArray();

// 2. Trending (System Generated Top 10, 20, 30, 40, 50)
        // Fetch 50 random songs for "Discovery" simulation
        $top50Songs = \App\Models\Songs::applyUserMood()
        ->orderByDesc('created_at')
            ->inRandomOrder()
            ->take(50)
            ->get();
 
        $trendingPlaylists = collect();
        foreach ([10, 20, 30, 40, 50] as $limit) {
            // Slice the collection for this limit
            $songsForPlaylist = $top50Songs->take($limit);
            
            // Skip if no songs available at all (optional, but good UX to hide empty or very low count?)
            // For now, allow it even if count < limit, it just shows real count.
            if ($songsForPlaylist->isEmpty()) {
                continue; 
            }

            $playlist = new Playlists();
            $playlist->id = "SYS-TOP-$limit";
            $playlist->name = "Top $limit Discovery";
            $playlist->description = "Randomly selected tracks for your $mood mood.";
            
            // Use the first song's album art as playlist cover
            $firstSong = $songsForPlaylist->first();
            $playlist->playlistPath = $firstSong ? $firstSong->photoPath : null; 
            
            // Set real relation for count()
            $playlist->setRelation('songs', $songsForPlaylist);
            
            // Calculate real duration
            $playlist->total_duration = $songsForPlaylist->sum('duration');
            
            $trendingPlaylists->push($playlist);
        }

        // 3. Random Mix (Totally random from remaining)
        // Note: System playlists don't use real IDs, so no need to exclude them from DB query.
        $randomPlaylists = (clone $queryBuilder)
            ->whereNotIn('id', $excludeIds)
            ->inRandomOrder()
            ->get();
            
        $username     = auth()->user()->username;
        $fullname     = auth()->user()->userDetail->fullname;
        $profilePhoto = auth()->user()->userDetail->profilePhoto;
        
        return view('main.index', compact('trendingPlaylists', 'newArrivalPlaylists', 'randomPlaylists', 'username', 'fullname', 'profilePhoto', 'mood', 'search'));
    }


    public function playlists(Request $request, $playlistId)
    {
        $search = $request->get('search');
        
        if (str_starts_with($playlistId, 'SYS-TOP-')) {
            // Handle System Playlist
            $limit = (int) str_replace('SYS-TOP-', '', $playlistId);
            
            // Create Mock Playlist
            $playlists = new Playlists();
            $playlists->id = $playlistId;
            $playlists->name = "Top $limit Discovery";
            $playlists->description = "The most popular tracks for your current mood.";
            $playlists->userId = "SYSTEM"; // System user
            $playlists->playlistPath = "https://ui-avatars.com/api/?name=Top+$limit&background=random&size=500";

            // Query Songs 
            // metric: withCount('playlists') -> orderBy desc
            // AND apply mood
            $mood = session('chooseMood', 'happy'); // Get current mood for query
            
            $query = \App\Models\Songs::query()->applyUserMood();
            
            if ($search) {
                $query->searchSongs($search);
            }

            // Query Songs 
            // metric: Discovery (Random + Latest Mix)
            // AND apply mood
            $mood = session('chooseMood', 'happy'); // Get current mood for query
            
            $query = \App\Models\Songs::query()->applyUserMood();
            
            if ($search) {
                $query->searchSongs($search);
            }

            // DISCOVERY MODE: Random Songs
            $songs = $query->inRandomOrder()
                           ->take($limit)
                           ->get();
                           
            // Set relation for view if needed
            $playlists->setRelation('songs', $songs);
            
        } else {
            // Normal Playlist
            $playlists = Playlists::findOrFail($playlistId);

            // 1. Get songs filtered by Mood ONLY (for Playlist Header Stats)
            $moodSongs = $playlists->songs()->applyUserMood()->get();
            // Manually set the relation so $playlists->total_duration uses this Filtered Collection
            $playlists->setRelation('songs', $moodSongs);

            // 2. Get songs for the List View (Apply Search if needed)
            $query = $playlists->songs()->applyUserMood();
            
            if ($search) {
                $query->searchSongs($search);
            }
            
            $songs = $query->get();
        }
        
        $mood  = session('chooseMood', 'happy');
        
        return view('main.playlists.index', compact('songs', 'mood', 'search', 'playlists'));
    }
}
