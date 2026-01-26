@props([
    'mood' => 'happy',
    'playlistName' => null,
    'playlistDesc' => null,
    'imageSource' => null,
    'totalSong' => 0,
    'totalTime' => 0
])


@php
    $bgBasedMood = [
        'happy' => 'bg-secondary-happy-100',
        'sad' => 'bg-secondary-sad-100',
        'relaxed' => 'bg-secondary-relaxed-100',
        'angry' => 'bg-secondary-angry-100',
    ];

    $bgSoftOpacityMood = [
        'happy' => 'bg-secondary-happy-10/90 hover:bg-secondary-happy-10',
        'sad' => 'bg-secondary-sad-10/90 hover:bg-secondary-sad-10',
        'relaxed' => 'bg-secondary-relaxed-10/90 hover:bg-secondary-relaxed-10',
        'angry' => 'bg-secondary-angry-10/90 hover:bg-secondary-angry-10'
    ];

    $textMood = [
        'happy' => 'text-secondary-happy-100',
        'sad' => 'text-secondary-sad-100',
        'relaxed' => 'text-secondary-relaxed-100',
        'angry' => 'text-secondary-angry-100'
    ];
@endphp


<div class='min-w-5 min-h-5 w-full h-max p-4 flex flex-row justify-start gap-3 rounded-md relative overflow-hidden duration-200 group {{ $bgSoftOpacityMood[$mood] }} md:w-lg'>
    <div class='h-26 w-26 shrink-0 relative z-100 rounded-sm object-cover overflow-hidden {{ $bgBasedMood[$mood] }} md:w-48 md:h-48'>
        <img src="{{ config('filesystems.disks.azure.url') . '/' . $imageSource }}" alt="{{ $playlistName }}" class='w-full h-full'>
    </div>
    <div class='font-secondaryAndButton flex flex-col justify-between relative z-1000 w-full'>
        <div>
            <p class='text-body-size font-bold {{ $textMood[$mood] }} md:text-paragraph'>{{ Str::upper($playlistName) }}</p>
            <p class='text-small text-primary-70 md:text-body-size'>{{ $playlistDesc }}</p>
        </div>
        <div class='flex flex-row h-max items-center gap-2 text-micro md:text-body-size'>
            <p class='text-primary-70 font-bold'>{{ $totalSong . ' ' . (($totalSong > 1) ? 'songs' : 'song') }}</p>
            <div class='{{ $bgBasedMood[$mood] }} rounded-full w-2 h-2'></div>
            <p class='text-primary-70'>{{ $totalTime }}</p>
        </div>
    </div>
    <img src="{{ asset('assets/moods/icons/' . $mood . '.png') }}" alt="$mood" class='w-44 h-44 opacity-10 absolute right-0 top-0 translate-x-12 -rotate-30 translate-y-4 group-hover:opacity-20 md:w-64 md:h-64 md:translate-x-20 md:translate-y-10'>
</div>