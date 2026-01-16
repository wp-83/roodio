@extends('layouts.main')


{{-- @section('title') --}}


@push('script')
    <script src="{{ asset('js/pages/audioControl.js') }}" defer></script>
@endpush


{{-- @section('mainContentContainerClass') --}}

@php
    // $mood = 'angry';
    $name = 'Thomas Aquinas Riald Prabadi';
    $username = 'Xullfikar831';

    $moodOptions = ['happy', 'sad', 'relaxed', 'angry'];

    $moodMessage = [
        'happy' => "Ahh!! You're happy now. Stay happy and keep smiling.",
        'sad' => "Oh no… you seem sad right now. It's okay, take your time.",
        'relaxed' => 'Hmm… you look relaxed now. Enjoy the calm and breathe easy.',
        'angry' => "Whoa… you seem angry right now. Take a deep breath, it'll pass."
    ];

    $textMoodStyle = [
        'happy' => 'text-secondary-happy-100',
        'sad' => 'text-secondary-sad-100',
        'relaxed' => 'text-secondary-relaxed-100',
        'angry' => 'text-secondary-angry-100'
    ];

    $bgMoodStyle = [
        'happy' => 'bg-secondary-happy-20',
        'sad' => 'bg-secondary-sad-20',
        'relaxed' => 'bg-secondary-relaxed-20',
        'angry' => 'bg-secondary-angry-20'
    ];

    $hoverBgMoodStyle = [
        'happy' => 'hover:bg-secondary-happy-20',
        'sad' => 'hover:bg-secondary-sad-20',
        'relaxed' => 'hover:bg-secondary-relaxed-20',
        'angry' => 'hover:bg-secondary-angry-20'
    ];
@endphp


@section('overlayContent')
    {{-- <form action=""></form> --}}

    {{-- <x-modal modalId='dayMood' additionalStyle='top-1/2 left-1/2 -translate-1/2 w-sm md:w-xl lg:w-2xl'>
        <x-slot name='header'>
            <p class='text-center font-bold text-primary-50 '>Welcome to ROODIO, Buddy!</p>
        </x-slot>
        <x-slot name='body'>
            <p class='my-3'>What do you feel today, Andi? Let us get to know you!</p>
            <div class='grid grid-cols-2 md:grid-cols-4'>
                @foreach ($moodOptions as $moodOption)
                    <div class='flex flex-col items-center justify-center group cursor-pointer hover:bg-shadedOfGray-10/30 rounded-lg animate-float-soft'>
                        <img src="{{ asset('assets/moods/' . Str::lower($moodOption) . '.png') }}" alt="{{ Str::lower($moodOption) }}" class='w-28 h-28 opacity-50 group-hover:opacity-100 md:w-36 md:h-36 lg:w-40 lg:h-40'>
                        <p class='w-fit mb-2 text-primary-70'>{{ Str::ucfirst($moodOption) }}</p>
                        <input type="radio" name="mood" id="{{ Str::lower($moodOption) . 'Mood' }}" value='{{ Str::lower($moodOption) }}' hidden>
                    </div>
                @endforeach
                
                
            </div>
        </x-slot>
    </x-modal> --}}

     {{-- <x-modal modalId='choosePlaylist' additionalStyle='top-1/2 left-1/2 -translate-1/2 w-xs md:w-sm'>
        <x-slot name='header'>
            <div class='w-full flex justify-center'>
                <img src="{{ asset('assets/moods/'. Str::lower($mood) .'.png') }}" alt="{{ Str::lower($mood) }}" class='w-40 h-40'>
            </div>
        </x-slot>
        <x-slot name='body'>
            <p class='text-justify'>{{ $moodMessage[$mood] . '' }}</p>
            <p class='mt-5'>Should I play something that matches your mood?</p>
            <div class='w-full flex flex-col '>
                <x-button content='Yes, Give me that' mood='{{ $mood }}'></x-button>
                <x-button content='No, Just random'></x-button>
            </div>
        </x-slot>
    </x-modal> --}}
    
    {{-- <x-modal modalId='profilePopup' additionalStyle='right-3 top-14 w-60 h-max '>
        <x-slot name='body'>
            <div class='absolute right-6 top-5' style='zoom: 0.75;'>
                <x-iconButton :mood='$mood' type='cross'></x-iconButton>
            </div>
            <div class='flex flex-col items-center gap-2'>
                <div class='w-20 h-20 rounded-full flex items-center justify-center {{ $bgMoodStyle[$mood] }}'>
                    <p class='text-title font-primary font-bold h-fit {{ $textMoodStyle[$mood] }}'>{{ Str::charAt(Str::upper($name), 0) }}</p>
                </div>
                <div class='flex flex-col items-center'>
                    <p class='text-small font-bold {{ $textMoodStyle[$mood] }}'>{{ Str::limit($name, 24) }}</p>
                    <p class='text-micro text-primary-60'>{{ '@' . Str::limit($username, 18) }}</p>
                </div>
            </div>
            <hr class='my-2 border-primary-50'>
            <div class='w-full flex flex-col gap-2.5 font-secondaryAndButton text-small'>
                <a href="">
                    <div class='h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 {{ $hoverBgMoodStyle[$mood] }}'>
                        <img src="{{ asset('assets/icons/user.svg') }}" alt="user" class='w-7 h-7'>
                        <p class='text-primary-60'>Edit Your Profile</p>
                    </div>
                </a>
                <a href="">
                    <div class='h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 {{ $hoverBgMoodStyle[$mood] }}'>
                        <img src="{{ asset('assets/icons/logout.svg') }}" alt="logout" class='w-7 h-7'>
                        <p class='text-primary-60'>Logout</p>
                    </div>
                </a>
            </div>
        </x-slot>
    </x-modal> --}}

    <x-modal modalId='changeMood' additionalStyle='right-48 top-14'>
        <x-slot name='body'>
            <p class='mb-3 font-bold text-primary-60'>Change Your Mood</p>
            <div class='w-full flex flex-col gap-2.5 font-secondaryAndButton text-small'>
                @foreach ($moodOptions as $moodOpt)
                    <a href="">
                        <div class='h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 {{ $hoverBgMoodStyle[$mood] }}'>
                            <img src="{{ asset('assets/moods/' . $moodOpt . '.png') }}" alt="{{ $moodOpt }}" class='w-7 h-7'>
                            <p class='text-primary-60'>{{ Str::ucfirst($moodOpt) }}</p>
                        </div>
                    </a>
                @endforeach
            </div>
        </x-slot>
    </x-modal>

@endsection


@section('mainContent')
    <div class='flex flex-row justify-content items-center'>
        <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="" class='h-42 w-42'>
        <div class='flex flex-col text-white'>
            <p class='font-primary text-white text-title font-bold'>Hi, Andi!</p>
            <p>Welcome to our life</p>
             {{-- @foreach ($playlists as $playlist)
                <p>Title: {{ $playlist->name }}</p>
            @endforeach --}}
        </div>
    </div>
    <div>
        <p class='text-title text-secondary-relaxed-30 font-primary font-bold mt-5'>Most Current Play Songs</p>
    </div>


@endsection


@section('bottomContent')
    <x-audioPlayer :mood='$mood'></x-audioPlayer>
@endsection
