@extends('layouts.main')


@php
    // definition of playlist data
    $playlistData = $songs->map(function($song) {
        return [
            'id' => $song->id,
            'title' => $song->title,
            'artist' => $song->artist,
            'cover' => !empty($song->photoPath) ? config('filesystems.disks.azure.url') . '/' . $song->photoPath : asset('assets/defaults/songCover.png'),
            'path' => config('filesystems.disks.azure.url') . '/' . $song->songPath,
            'lyrics' => $song->lyrics ?? '',
        ];
    });

    $textMoodStyle = [
        'happy' => 'text-secondary-happy-30',
        'sad' => 'text-secondary-sad-30',
        'relaxed' => 'text-secondary-relaxed-30',
        'angry' => 'text-secondary-angry-30'
    ];

    $activeRowStyle = [
        'happy' => 'bg-secondary-happy-10/30',
        'sad' => 'bg-secondary-sad-10/30',
        'relaxed' => 'bg-secondary-relaxed-10/30',
        'angry' => 'bg-secondary-angry-10/30'
    ];
@endphp


<script>
    // give data to global window
    window.currentPlaylist = @json($playlistData);
    // Notify audioControl.js to re-sync playlist
    window.dispatchEvent(new CustomEvent('playlist-updated'));
</script>


@section('mainContent')
    <div class='flex flex-row gap-2 items-center mb-8'>
        <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="{{ $mood }}" class='w-32'>
        <div class='flex flex-col'>
            <p class='font-primary {{ $textMoodStyle[$mood] }} font-bold text-subtitle lg:text-title'>{{ $playlists->name ?? 'Playlist Title' }}</p>
            <div class='flex flex-col gap-1.5 items-start'>
                <p class='font-secondaryAndButton text-white text-small lg:text-body-size'>{{ $playlists->description ?? 'Playlist Desc' }}</p>
                <p class='font-secondaryAndButton text-white text-micro'>{{ $playlists->total_duration ?? '0 min' }}</p>
            </div> 
        </div>
    </div>
    <div class='w-full flex flex-col gap-3'>
        @foreach ($songs as $index => $song)
            <button onclick="playByIndex({{ $index }})" class="text-left group/btn" id="song-{{ $index }}" data-active-class="{{ $activeRowStyle[$mood] }}">
                <x-songCard :mood="$mood" :photoPath="$song->photoPath" :title="$song->title" :artist="$song->artist" :durationFormatted="$song->duration_formatted" :datePublished="$song->datePublished->diffForHumans()"></x-songCard>
            </button>
        @endforeach
    </div>
@endsection
