@props([
    'mood',
    'photoPath' => null,
    'title',
    'artist',
    'durationFormatted',
    'datePublished',
])


@php
    $hoverPlayStyle = [
        'happy' => 'bg-secondary-happy-30/50',
        'sad' => 'bg-secondary-sad-30/50',
        'relaxed' => 'bg-secondary-relaxed-30/50',
        'angry' => 'bg-secondary-angry-30/50'
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
@endphp


<div class='min-h-2 w-full h-max p-3 flex flex-row gap-3 hover:bg-shadedOfGray-10/10 rounded-md group cursor-pointer transition-colors duration-200'>
    <div class='w-20 h-20 bg-shadedOfGray-10 rounded-md overflow-hidden relative'>
        @if (empty($photoPath))
            <img src="{{ asset('assets/defaults/songCover.png') }}" alt="{{ $mood }}" class='w-full h-full object-cover relative'>
        @else
            <img src="{{ config('filesystems.disks.azure.url') . '/' . $photoPath }}" alt="{{ $title }}" class='w-full h-full object-cover relative'>
        @endif
        <div class='{{ $hoverPlayStyle[$mood] }} w-full h-full absolute top-0 left-0 invisible flex items-center justify-center group-hover:visible'>
            <img src="{{ asset('assets/icons/play-dark.svg') }}" alt="play-btn" class='w-9 h-9 opacity-80'>
        </div>
    </div>
    <div class='flex flex-col justify-between font-secondaryAndButton'>
        <div>
            <p class='{{ $textMoodCustomStyle[$mood] }} text-body-size font-bold'>{{ $title }}</p>
            <div class='flex flex-row gap-1.5 items-center text-white'>
                <p class='text-small'>{{ $artist }}</p>
                <div class='w-1.5 h-1.5 rounded-full {{ $roundedMoodStyle[$mood] }}'></div>
                <p class='text-small'>{{ $durationFormatted }}</p>
            </div>
        </div>
        <p class='text-shadedOfGray-20 text-micro'>{{ $datePublished ? $datePublished : 'Unknown date' }}</p>
    </div>
</div>