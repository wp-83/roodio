<?php

namespace Database\Seeders;

use App\Models\Playlists;
use App\Models\Songs;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class PlaylistsSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // User ID for seeding (adjust if needed)
        $userId = 'US-0000001';

        // Get all songs grouped by mood
        $angrySongs = Songs::where('moodId', 'MD-0000004')->pluck('id')->toArray();
        $happySongs = Songs::where('moodId', 'MD-0000001')->pluck('id')->toArray();
        $relaxedSongs = Songs::where('moodId', 'MD-0000003')->pluck('id')->toArray();
        $sadSongs = Songs::where('moodId', 'MD-0000002')->pluck('id')->toArray();
        $allSongs = Songs::pluck('id')->toArray();

        // Define playlists with their songs
        $playlists = [
            // COMPLETE MOOD PLAYLISTS (FOR TESTING - ALL SONGS PER MOOD)
            [
                'name' => 'All Angry Songs',
                'description' => 'Complete collection of all angry mood songs',
                'playlistPath' => 'playlists/all-angry.jpg',
                'songs' => $angrySongs, // ALL 50 angry songs
            ],
            [
                'name' => 'All Happy Songs',
                'description' => 'Complete collection of all happy mood songs',
                'playlistPath' => 'playlists/all-happy.jpg',
                'songs' => $happySongs, // ALL 50 happy songs
            ],
            [
                'name' => 'All Relaxed Songs',
                'description' => 'Complete collection of all relaxed mood songs',
                'playlistPath' => 'playlists/all-relaxed.jpg',
                'songs' => $relaxedSongs, // ALL 50 relaxed songs
            ],
            [
                'name' => 'All Sad Songs',
                'description' => 'Complete collection of all sad mood songs',
                'playlistPath' => 'playlists/all-sad.jpg',
                'songs' => $sadSongs, // ALL 50 sad songs
            ],

            // ANGRY MOOD PLAYLISTS
            [
                'name' => 'Angry Anthems',
                'description' => 'Let it all out with these powerful angry tracks',
                'playlistPath' => 'playlists/angry-anthems.jpg',
                'songs' => array_slice($angrySongs, 0, 20),
            ],
            [
                'name' => 'Breakup Rage',
                'description' => 'Perfect soundtrack for when you\'re done with relationships',
                'playlistPath' => 'playlists/breakup-rage.jpg',
                'songs' => array_slice($angrySongs, 10, 15),
            ],
            [
                'name' => 'Workout Fury',
                'description' => 'Channel your anger into your workout',
                'playlistPath' => 'playlists/workout-fury.jpg',
                'songs' => array_slice($angrySongs, 20, 18),
            ],
            [
                'name' => 'Emotional Release',
                'description' => 'Songs for when you need to let out your emotions',
                'playlistPath' => 'playlists/emotional-release.jpg',
                'songs' => array_slice($angrySongs, 5, 25),
            ],

            // HAPPY MOOD PLAYLISTS
            [
                'name' => 'Feel Good Hits',
                'description' => 'Uplifting songs to brighten your day',
                'playlistPath' => 'playlists/feel-good-hits.jpg',
                'songs' => array_slice($happySongs, 0, 25),
            ],
            [
                'name' => 'Party All Night',
                'description' => 'Turn up the volume and dance all night long',
                'playlistPath' => 'playlists/party-all-night.jpg',
                'songs' => array_slice($happySongs, 15, 20),
            ],
            [
                'name' => 'Summer Vibes',
                'description' => 'Perfect songs for sunny summer days',
                'playlistPath' => 'playlists/summer-vibes.jpg',
                'songs' => array_slice($happySongs, 10, 22),
            ],
            [
                'name' => 'Morning Energy',
                'description' => 'Start your day with positive energy',
                'playlistPath' => 'playlists/morning-energy.jpg',
                'songs' => array_slice($happySongs, 5, 18),
            ],
            [
                'name' => 'Celebration Mix',
                'description' => 'Celebrate life with these joyful tunes',
                'playlistPath' => 'playlists/celebration-mix.jpg',
                'songs' => array_slice($happySongs, 25, 15),
            ],
            [
                'name' => 'Road Trip Bangers',
                'description' => 'Your ultimate road trip companion',
                'playlistPath' => 'playlists/road-trip.jpg',
                'songs' => array_slice($happySongs, 30, 20),
            ],

            // RELAXED MOOD PLAYLISTS
            [
                'name' => 'Chill Vibes',
                'description' => 'Relax and unwind with these mellow tracks',
                'playlistPath' => 'playlists/chill-vibes.jpg',
                'songs' => array_slice($relaxedSongs, 0, 20),
            ],
            [
                'name' => 'Study Focus',
                'description' => 'Stay focused with calming background music',
                'playlistPath' => 'playlists/study-focus.jpg',
                'songs' => array_slice($relaxedSongs, 10, 18),
            ],
            [
                'name' => 'Coffee Shop Ambience',
                'description' => 'Like sitting in your favorite coffee shop',
                'playlistPath' => 'playlists/coffee-shop.jpg',
                'songs' => array_slice($relaxedSongs, 5, 22),
            ],
            [
                'name' => 'Late Night Thoughts',
                'description' => 'For those quiet, contemplative nights',
                'playlistPath' => 'playlists/late-night.jpg',
                'songs' => array_slice($relaxedSongs, 15, 20),
            ],
            [
                'name' => 'Sunset Sessions',
                'description' => 'Wind down as the day comes to an end',
                'playlistPath' => 'playlists/sunset-sessions.jpg',
                'songs' => array_slice($relaxedSongs, 20, 15),
            ],
            [
                'name' => 'Meditation & Mindfulness',
                'description' => 'Find your inner peace',
                'playlistPath' => 'playlists/meditation.jpg',
                'songs' => array_slice($relaxedSongs, 8, 17),
            ],

            // SAD MOOD PLAYLISTS
            [
                'name' => 'Heartbreak Hotel',
                'description' => 'Songs for when your heart is broken',
                'playlistPath' => 'playlists/heartbreak.jpg',
                'songs' => array_slice($sadSongs, 0, 20),
            ],
            [
                'name' => 'Melancholic Moments',
                'description' => 'Embrace the sadness with these emotional tracks',
                'playlistPath' => 'playlists/melancholic.jpg',
                'songs' => array_slice($sadSongs, 10, 18),
            ],
            [
                'name' => 'Crying in the Rain',
                'description' => 'Let the tears flow freely',
                'playlistPath' => 'playlists/crying-rain.jpg',
                'songs' => array_slice($sadSongs, 15, 22),
            ],
            [
                'name' => 'Lost Love',
                'description' => 'Remembering what once was',
                'playlistPath' => 'playlists/lost-love.jpg',
                'songs' => array_slice($sadSongs, 5, 20),
            ],
            [
                'name' => 'Lonely Nights',
                'description' => 'Company for your solitude',
                'playlistPath' => 'playlists/lonely-nights.jpg',
                'songs' => array_slice($sadSongs, 20, 15),
            ],
            [
                'name' => 'Beautiful Sadness',
                'description' => 'Finding beauty in the pain',
                'playlistPath' => 'playlists/beautiful-sadness.jpg',
                'songs' => array_slice($sadSongs, 8, 19),
            ],

            // MIXED MOOD PLAYLISTS
            [
                'name' => 'Emotional Rollercoaster',
                'description' => 'A journey through all emotions',
                'playlistPath' => 'playlists/emotional-rollercoaster.jpg',
                'songs' => array_merge(
                    array_slice($happySongs, 0, 5),
                    array_slice($sadSongs, 0, 5),
                    array_slice($angrySongs, 0, 5),
                    array_slice($relaxedSongs, 0, 5)
                ),
            ],
            [
                'name' => 'Top 50 Hits',
                'description' => 'The most popular songs across all moods',
                'playlistPath' => 'playlists/top-50.jpg',
                'songs' => array_slice($allSongs, 0, 50),
            ],
            [
                'name' => 'Weekly Discovery',
                'description' => 'Fresh picks for this week',
                'playlistPath' => 'playlists/weekly-discovery.jpg',
                'songs' => array_merge(
                    array_slice($happySongs, 5, 8),
                    array_slice($relaxedSongs, 5, 8),
                    array_slice($sadSongs, 3, 6),
                    array_slice($angrySongs, 3, 6)
                ),
            ],
            [
                'name' => '2010s Throwback',
                'description' => 'Nostalgia from the 2010s era',
                'playlistPath' => 'playlists/2010s-throwback.jpg',
                'songs' => array_merge(
                    array_slice($happySongs, 0, 10),
                    array_slice($sadSongs, 0, 10),
                    array_slice($relaxedSongs, 0, 5)
                ),
            ],
            [
                'name' => 'Acoustic Sessions',
                'description' => 'Stripped down and beautiful',
                'playlistPath' => 'playlists/acoustic.jpg',
                'songs' => array_merge(
                    array_slice($relaxedSongs, 10, 10),
                    array_slice($sadSongs, 10, 10)
                ),
            ],
            [
                'name' => 'Power Ballads',
                'description' => 'Epic emotional anthems',
                'playlistPath' => 'playlists/power-ballads.jpg',
                'songs' => array_merge(
                    array_slice($sadSongs, 5, 8),
                    array_slice($angrySongs, 5, 7),
                    array_slice($happySongs, 10, 5)
                ),
            ],
            [
                'name' => 'Indie Favorites',
                'description' => 'Best of indie music',
                'playlistPath' => 'playlists/indie-favorites.jpg',
                'songs' => array_merge(
                    array_slice($relaxedSongs, 15, 10),
                    array_slice($sadSongs, 15, 8)
                ),
            ],
            [
                'name' => 'Love Songs Collection',
                'description' => 'Songs about love in all its forms',
                'playlistPath' => 'playlists/love-songs.jpg',
                'songs' => array_merge(
                    array_slice($happySongs, 20, 8),
                    array_slice($sadSongs, 20, 8),
                    array_slice($relaxedSongs, 20, 6)
                ),
            ],
        ];

        // Create playlists and attach songs
        foreach ($playlists as $playlistData) {
            // Create playlist
            $playlist = Playlists::create([
                'userId' => $userId,
                'name' => $playlistData['name'],
                'description' => $playlistData['description'],
                'playlistPath' => $playlistData['playlistPath'],
            ]);

            // Attach songs to playlist using the Tracks table
            foreach ($playlistData['songs'] as $index => $songId) {
                DB::table('tracks')->insert([
                    'id' => $this->generateTrackId($index),
                    'playlistId' => $playlist->id,
                    'songId' => $songId,
                    'created_at' => now(),
                    'updated_at' => now(),
                ]);
            }
        }
    }

    /**
     * Generate unique track ID
     */
    private function generateTrackId($index)
    {
        $lastId = DB::table('tracks')->orderBy('id', 'desc')->value('id');
        
        if (!$lastId) {
            $number = 1;
        } else {
            $number = (int) substr($lastId, 3) + 1;
        }
        
        return 'TR-' . str_pad($number, 7, '0', STR_PAD_LEFT);
    }
}
