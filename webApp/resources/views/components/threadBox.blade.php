@props([
    'mood',
    'creator',
    'createdAt',
    'title',
    'content',
    'threadId',
    'profilePicture'
])

@php
    $bgContainer = [
        'happy' => 'bg-secondary-happy-10',
        'sad' => 'bg-secondary-sad-10',
        'relaxed' => 'bg-secondary-relaxed-10',
        'angry' => 'bg-secondary-angry-10',
    ];

    $borderMood = [
        'happy' => 'border-secondary-happy-100',
        'sad' => 'border-secondary-sad-100',
        'relaxed' => 'border-secondary-relaxed-100',
        'angry' => 'border-secondary-angry-100'
    ];

    $textMood = [
        'happy' => 'text-secondary-happy-100',
        'sad' => 'text-secondary-sad-100',
        'relaxed' => 'text-secondary-relaxed-100',
        'angry' => 'text-secondary-angry-100',
    ];
@endphp

<div class='{{ $bgContainer[$mood] }} rounded-lg h-max p-5 w-full'>
    <div class='flex flex-row items-center gap-2 w-full'>
        <div class='w-18 h-18 border-2 {{ $borderMood[$mood] }} rounded-full flex items-center justify-center'>
            <div class='w-16 h-16 bg-primary-10 rounded-full flex items-center justify-center relative z-5 overflow-hidden'>
            @if (!empty($profilePicture))
                <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePicture }}" alt="{{ $creator }}" class='w-full h-full object-cover'> 
            @else
                <p class='text-subtitle font-primary font-bold text-primary-70 h-fit'>{{ Str::charAt(Str::upper($creator), 0) }}</p>
            @endif
        </div>
        </div>
        <div class='flex-1 w-full font-secondaryAndButton'>
            <div class='flex flex-col'>
                <p class='text-body-size font-bold {{ $textMood[$mood] }}'>{{ $creator }}</p>
                <p class='text-micro lg:text-small'>{{ $createdAt }}</p>
            </div>
            <div>
                {{-- <x-button mood='grayscale'></x-button> --}}
            </div>
        </div>
    </div>
    <hr class='my-3'>
    <div>
        <p class='font-bold font-primary text-paragraph lg:text-subtitle '>{{ $title }}</p>
        <p class='font-secondaryAndButton text-small lg:text-body-size'>{{ $content }}</p>
    </div>
    <div class='flex flex-row '>
        <livewire:user.reaction-button :thread-id="$threadId" />
    </div>
</div>