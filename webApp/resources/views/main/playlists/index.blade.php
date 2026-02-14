@extends('layouts.main')


@php
    // definition of playlist data
    $playlistData = $songs->map(function($song) {
        return [
            'id' => $song->id,
            'title' => $song->title,
            'artist' => $song->artist,
            'cover' => $song->cover_path ?? asset('assets/images/default-music.png'), // default image
            'path' => config('filesystems.disks.azure.url') . '/' . $song->songPath
        ];
    });

    $textMoodStyle = [
        'happy' => 'text-secondary-happy-30',
        'sad' => 'text-secondary-sad-30',
        'relaxed' => 'text-secondary-relaxed-30',
        'angry' => 'text-secondary-angry-30'
    ];

    $textMoodCustomStyle = [
        'happy' => 'text-secondary-happy-50',
        'sad' => 'text-secondary-sad-50',
        'relaxed' => 'text-secondary-relaxed-50',
        'angry' => 'text-secondary-angry-50'
    ];

    $roundedMoodStyle = [
        'happy' => 'bg-secondary-happy-85',
        'sad' => 'bg-secondary-sad-85',
        'relaxed' => 'bg-secondary-relaxed-85',
        'angry' => 'bg-secondary-angry-85'
    ];

    $hoverPlayStyle = [
        'happy' => 'bg-secondary-happy-30/50',
        'sad' => 'bg-secondary-sad-30/50',
        'relaxed' => 'bg-secondary-relaxed-30/50',
        'angry' => 'bg-secondary-angry-30/50'
    ];
@endphp


<script>
    // give data to global window
    window.currentPlaylist = @json($playlistData);
</script>


@section('mainContent')
    <div class='flex flex-row gap-2 items-center mb-8'>
        <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="{{ $mood }}" class='w-32'>
        <div class='flex flex-col'>
            <p class='font-primary {{ $textMoodStyle[$mood] }} font-bold text-subtitle lg:text-title'>Playlist Title</p>
            <p class='font-secondaryAndButton text-white text-small lg:text-body-size'>Playlist Desc</p>
        </div>
    </div>
    <div class='w-full flex flex-col gap-3'>
        @foreach ($songs as $index => $song)
            <button onclick="playByIndex({{ $index }})" class="text-left">
                <div class='min-h-2 w-full h-max p-3 flex flex-row gap-3 hover:bg-shadedOfGray-10/10 rounded-md group cursor-pointer'>
                    <div class='w-20 h-20 bg-shadedOfGray-10 rounded-md overflow-hidden relative'>
                        <img src="{{ config('filesystems.disks.azure.url') . '/' . $song->photoPath }}" alt="{{ $song->title }}" class='w-full h-full object-cover relative'>
                        <div class='{{ $hoverPlayStyle[$mood] }} w-full h-full absolute top-0 left-0 invisible flex items-center justify-center group-hover:visible'>
                            <img src="{{ asset('assets/icons/play-dark.svg') }}" alt="play-btn" class='w-9 h-9 opacity-80'>
                        </div>
                    </div>
                    <div class='flex flex-col justify-between font-secondaryAndButton'>
                        <div>
                            <p class='{{ $textMoodCustomStyle[$mood] }} text-body-size font-bold'>{{ $song->title }}</p>
                            <div class='flex flex-row gap-1.5 items-center text-white'>
                                <p class='text-small'>{{ $song->artist }}</p>
                                <div class='w-1.5 h-1.5 rounded-full {{ $roundedMoodStyle[$mood] }}'></div>
                                <p class='text-small'>Played</p>
                            </div>
                        </div>
                        <p class='text-shadedOfGray-20 text-micro'>Release date</p>
                    </div>
                </div>
            </button>
        @endforeach
    </div>
@endsection
