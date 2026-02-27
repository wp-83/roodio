@props([
    'mood',
    'fullname',
    'username',
    'followingCount',
    'followerCount',
    'user',
    'mainUser',
    'createdAt',
    'profilePhoto' => null
])


@php
    $bgPhotoStyle = [
        'happy' => 'bg-secondary-happy-30',
        'sad' => 'bg-secondary-sad-30',
        'relaxed' => 'bg-secondary-relaxed-30',
        'angry' => 'bg-secondary-angry-30'
    ];

    $textMoodStyle = [
        'happy' => 'text-secondary-happy-50',
        'sad' => 'text-secondary-sad-50',
        'relaxed' => 'text-secondary-relaxed-50',
        'angry' => 'text-secondary-angry-50'
    ];

    $pipeMoodStyle = [
        'happy' => 'text-secondary-happy-100',
        'sad' => 'text-secondary-sad-100',
        'relaxed' => 'text-secondary-relaxed-100',
        'angry' => 'text-secondary-angry-100'
    ];

    $bgBasedMood = [
        'happy' => 'bg-secondary-happy-100',
        'sad' => 'bg-secondary-sad-100',
        'relaxed' => 'bg-secondary-relaxed-100',
        'angry' => 'bg-secondary-angry-100',
    ];

    $accountDurationStyle = [
        'happy' => 'bg-secondary-happy-100',
        'sad' => 'bg-secondary-sad-100',
        'relaxed' => 'bg-secondary-relaxed-100',
        'angry' => 'bg-secondary-angry-100'
    ];
@endphp


<div class='flex md:hidden w-full h-max text-white py-3 px-4 flex-row items-center rounded-md overflow-hidden font-secondaryAndButton hover:bg-white/15'>
    <div class='mr-2.5'>
        <div class='w-30 h-30 overflow-hidden {{ $bgPhotoStyle[$mood] }}' style="clip-path: polygon(50% 0%, 83% 12%, 100% 43%, 94% 78%, 68% 100%, 32% 100%, 6% 78%, 0% 43%, 17% 12%);">
            @if (!empty($profilePhoto))
                <img src="{{ config('filesystems.storage_url') . '/' . $profilePhoto }}" alt="{{ $fullname }}" class='w-full h-full object-cover opacity-80'>
            @else
                <img src="{{ asset('assets/defaults/user.jpg') }}" alt="userDefault" class='w-full h-full object-cover opacity-75'>
            @endif
        </div>
    </div>
    <div class='flex flex-col gap-3'>
        <div class='flex flex-col items-start'>
            <p class='text-body-size font-bold {{ $textMoodStyle[$mood] }}'>{{ Str::limit($fullname, 18) }}</p>
            <p class='text-small italic'>{{ '@' . $username }}</p>
        </div>
        <div class='flex flex-col gap-1'>
            <div class='flex flex-row items-center justify-center gap-2 text-micro'>
                <h1 class="">{{ $followerCount . ' Follower' . (($followerCount > 1) ? 's' : '') }}</h1>
                <div class='{{ $bgBasedMood[$mood] }} rounded-full w-2 h-2'></div>
                <h1 class="">{{ $followingCount . ' Following' }}</h1>
            </div>
            @if ($user->id !== $mainUser->id)
                <livewire:user.button-follow
                    :userId="$user->id"
                    :mood="$mood"
                    customClass="w-max"
                    :wire:key="'mobile-follow-'.$user->id"
                />
            @endif

        </div>
    </div>
</div>

<div class='hidden md:block relative w-full min-h-90 rounded-lg overflow-hidden group'>
    @if (!empty($profilePhoto))
        <img src="{{ config('filesystems.storage_url') . '/' . $profilePhoto }}" alt="{{ $username }}" class='w-full h-full object-cover opacity-80'>
    @else
        <img src="{{ asset('assets/defaults/user.jpg') }}" alt="userDefault" class='w-full h-full object-cover opacity-75'>
    @endif
    <div class='absolute flex flex-col gap-2 w-full h-2/5 bg-linear-to-t from-shadedOfGray-85 to-white/30 bottom-0 left-0 group-hover:border-t-4 group-hover:border-white duration-150' style="clip-path: polygon(0% 38%, 45% 38%, 55% 0%, 100% 0%, 100% 100%, 0% 100%);">
        <div class='w-full h-max flex justify-end' style="zoom:0.775;">
             @if ($user->id !== $mainUser->id)
                <livewire:user.button-follow
                    :userId="$user->id"
                    :mood="$mood"
                    customClass="w-max mr-3 mt-3.5"
                    :wire:key="'desktop-follow-'.$user->id"
                />
            @endif
        </div>
        <div class='w-full flex flex-col gap-1 items-start font-secondaryAndButton text-white px-3 pt-2'>
            <div class='flex flex-row items-center gap-2 w-full'>
                <p class='font-bold {{ $textMoodStyle[$mood] }} text-body-size'>{{ Str::limit($fullname, 13) }}</p>
                <p class='font-bold {{ $pipeMoodStyle[$mood] }}'>|</p>
                <p class='text-small italic'>{{ '@' . Str::limit($username, 15) }}</p>
            </div>
            <div class='flex flex-row items-center justify-center gap-2 text-micro'>
                <p>{{ $followerCount . ' Follower' . (($followerCount > 1) ? 's' : '') }}</p>
                <div class='{{ $bgBasedMood[$mood] }} rounded-full w-2 h-2'></div>
                <p>{{ $followingCount . ' Following' }}</p>
            </div>
        </div>
        <div class='w-28 h-28 absolute -rotate-20 -right-6 -bottom-10 opacity-30'>
            <img src="{{ asset('assets/moods/icons/' . $mood . '.png') }}" alt="{{ $mood }}" class='w-max h-max'>
        </div>
    </div>
    <div class='absolute top-2 left-2 {{ $accountDurationStyle[$mood] }} w-max px-3 py-1 rounded-full'>
        <p class='text-white'>{{ $createdAt . ' year' . (($createdAt >= 2) ? 's' : '') }}</p>
    </div>
</div>
