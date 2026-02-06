@extends('layouts.main')


@section('title', 'ROODIO - Social')


@push('head')
    <meta name="csrf-token" content="{{ csrf_token() }}">
@endpush


@section('mainContentContainerClass')


@php
    $mainUser = auth()->user();

    $bgPhotoStyle = [
        'happy' => 'bg-secondary-happy-30',
        'sad' => 'bg-secondary-sad-30',
        'relaxed' => 'bg-secondary-relaxed-30',
        'angry' => 'bg-secondary-angry-30'
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

    $textMoodStyle = [
        'happy' => 'text-secondary-happy-50',
        'sad' => 'text-secondary-sad-50',
        'relaxed' => 'text-secondary-relaxed-50',
        'angry' => 'text-secondary-angry-50'
    ];

    $accountDurationStyle = [
        'happy' => 'bg-secondary-happy-100',
        'sad' => 'bg-secondary-sad-100',
        'relaxed' => 'bg-secondary-relaxed-100',
        'angry' => 'bg-secondary-angry-100'
    ];

    $textColor = [
        'happy' => 'text-secondary-happy-30',
        'sad' => 'text-secondary-sad-30',
        'relaxed' => 'text-secondary-relaxed-30',
        'angry' => 'text-secondary-angry-30'
    ];
@endphp


@section('mainContent')
    <div class='mb-5 md:mb-9 w-max'>
        <div class='flex flex-row gap-3 items-center justify-center'>
            <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="{{ $mood }}" class='hidden md:inline h-40 w-40'>
            <div class='flex flex-col'>
                <div class='w-0 relative overflow-hidden typingTextAnimation max-w-max '>
                    <p class='font-primary text-title font-bold {{ $textColor[$mood] }} md:text-hero' >Socials</p>
                </div>
                <p class='font-secondaryAndButton text-white text-justify contentFadeLoad text-small md:text-body-size'>Hi, {{ Str::before($mainUser->userDetail->fullname, ' ') }}! Let's give some social space to others.</p>
            </div>
        </div>
    </div>

    <form action="{{ route('social.index') }}" method="GET">
        <div class='mb-7 flex flex-row gap-3 w-full lg:justify-end contentFadeLoad'>
            <x-filterButton id='all' name='filter' value='all' :mood='$mood' label='All Users' onchange="this.form.submit()"></x-filterButton>
            <x-filterButton id='following' name='filter' value='following' :mood='$mood' label='Following Only' onchange="this.form.submit()"></x-filterButton>
        </div>
    </form>

    @if (count($users) > 0)
        <div class="w-full mt-4 grid grid-cols-1 justify-items-center md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-7 md:gap-9 contentFadeLoad">        
            @foreach($users as $user)
                @php
                    $username = $user->username;
                    $fullname = $user->userDetail->fullname;
                    $profilePhoto = $user->userDetail->profilePhoto;
                    $createdAt = floor($user->userDetail->created_at->diffInYears(now()));
                    $followerCount = $user->followings()->count();
                    $followingCount = $user->followers()->count();
                    $checkSelfProfile = !($user->id == $mainUser->id);
                @endphp
                @if ($checkSelfProfile)
                    <div class='w-full h-max text-white py-3 px-4 flex flex-row items-center rounded-md overflow-hidden font-secondaryAndButton hover:bg-white/15 md:hidden'>
                        <div class='mr-2.5'>
                            <div class='w-30 h-30 overflow-hidden {{ $bgPhotoStyle[$mood] }}' style="clip-path: polygon(50% 0%, 83% 12%, 100% 43%, 94% 78%, 68% 100%, 32% 100%, 6% 78%, 0% 43%, 17% 12%);">
                                @if (!empty($profilePhoto))
                                    <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" alt="{{ $fullname }}" class='w-full h-full object-cover opacity-80'>
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
                            <div class='flex flex-col'>
                                <div class='flex flex-row items-center justify-center gap-2 text-micro'>
                                    <h1 class="">{{ $user->followings()->count() . ' Follower' . (($user->followings()->count() > 1) ? 's' : '') }}</h1>
                                    <div class='{{ $bgBasedMood[$mood] }} rounded-full w-2 h-2'></div>
                                    <h1 class="">{{ $user->followers()->count() . ' Following' }}</h1>
                                </div>
                                <x-button :mood='$mood' content='Following' class='w-max px-2' style="zoom:0.85;"></x-button>
                            </div>
                        </div>
                    </div>
                    
                    <div class='relative w-full h-84 rounded-lg overflow-hidden hidden group md:inline'>
                        @if (!empty($profilePhoto))
                            <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" alt="{{ $username }}" class='w-full h-full object-cover opacity-80'>
                        @else
                            <img src="{{ asset('assets/defaults/user.jpg') }}" alt="userDefault" class='w-full h-full object-cover opacity-75'>
                        @endif
                        <div class='absolute flex flex-col gap-2 w-full h-2/5 bg-linear-to-t from-shadedOfGray-85 to-white/30 bottom-0 left-0 group-hover:border-t-4 group-hover:border-white duration-150' style="clip-path: polygon(0% 38%, 45% 38%, 55% 0%, 100% 0%, 100% 100%, 0% 100%);">
                            <div class='w-full h-max flex justify-end' style="zoom:0.725;">
                                <x-button content='Following' :mood='$mood' class='w-max mr-3 mt-3.5'></x-button>
                            </div>
                            <div class='w-full flex flex-col gap-1 items-start font-secondaryAndButton text-white px-3 pt-2'>
                                <div class='flex flex-row items-center gap-2 w-full'>
                                    <p class='font-bold {{ $textMoodStyle[$mood] }} text-body-size'>{{ Str::limit($fullname, 17) }}</p>
                                    <p class='font-bold {{ $pipeMoodStyle[$mood] }}'>|</p>
                                    <p class='text-small italic'>{{ '@' . $username }}</p>
                                </div>
                                <div class='flex flex-row items-center justify-center gap-2 text-micro'>
                                    <p>{{ $followingCount . ' Follower' . (($followingCount > 1) ? 's' : '') }}</p>
                                    <div class='{{ $bgBasedMood[$mood] }} rounded-full w-2 h-2'></div>
                                    <p>{{ $followingCount . ' Following' }}</p>
                                </div>
                            </div>
                            <div class='w-28 h-28 absolute -rotate-20 -right-6 -bottom-10 opacity-30'>
                                <img src="{{ asset('assets/moods/icons/' . $mood . '.png') }}" alt="" class='w-max h-max'>
                            </div>
                        </div>
                        <div class='absolute top-2 left-2 {{ $accountDurationStyle[$mood] }} w-max px-3 py-1 rounded-full'>
                            <p class='text-white'>{{ $createdAt . ' year' . (($createdAt >= 2) ? 's' : '') }}</p>
                        </div>
                    </div>
                @endif
            @endforeach
        </div>
    @else
        <p class='w-full text-white font-secondaryAndButton text-small lg:text-body-size'>There is no other user(s) can be displayed.</p>
    @endif
@endsection
