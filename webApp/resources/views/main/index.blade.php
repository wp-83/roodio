@extends('layouts.main')


{{-- @section('title') --}}


@push('script')
    <script src="{{ asset('js/pages/audioControl.js') }}" defer></script>
    <script src="{{ asset('js/pages/main/index.js') }}" defer></script>
@endpush


{{-- @section('mainContentContainerClass') --}}

@php
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

    $checkboxStyle = [
        'happy' => 'accent-secondary-happy-50',
        'sad' => 'accent-secondary-sad-50',
        'relaxed' => 'accent-secondary-relaxed-50',
        'angry' => 'accent-secondary-angry-50'
    ];
@endphp


@section('overlayContent')
    @if (!session()->has('chooseMood'))
        <x-modal modalId='dayMood' additionalStyle='top-1/2 left-1/2 -translate-1/2 w-sm md:w-xl lg:w-2xl'>
            <x-slot name='header'>
                <p class='text-center font-bold text-primary-50 '>Welcome to ROODIO, Buddy!</p>
            </x-slot>
            <x-slot name='body'>
                <p class='my-3'>What do you feel today, Andi? Let us get to know you!</p>

                <form action="{{ route('mood.store') }}" method="POST">
                    @csrf
                    <div class='grid grid-cols-2 md:grid-cols-4'>
                        @foreach ($moodOptions as $moodOption)
                            <label class='flex flex-col items-center justify-center group cursor-pointer hover:bg-shadedOfGray-10/30 rounded-lg animate-float-soft'>

                                <img src="{{ asset('assets/moods/' . Str::lower($moodOption) . '.png') }}"
                                    alt="{{ Str::lower($moodOption) }}"
                                    class='w-28 h-28 opacity-50 group-hover:opacity-100 md:w-36 md:h-36 lg:w-40 lg:h-40'>

                                <p class='w-fit mb-2 text-primary-70'>{{ Str::ucfirst($moodOption) }}</p>

                                <input type="radio"
                                    name="mood"
                                    value='{{ Str::lower($moodOption) }}'
                                    class="hidden"
                                    onchange="this.form.submit()">
                            </label>
                        @endforeach
                    </div>
                </form>

            </x-slot>
        </x-modal>
    @endif

    @if (session()->has('chooseMood') && !session()->has('preferenceMood'))
        <x-modal modalId='choosePlaylist' additionalStyle='top-1/2 left-1/2 -translate-1/2 w-xs md:w-sm'>
            <x-slot name='header'>
                <div class='w-full flex justify-center'>
                    <img src="{{ asset('assets/moods/'. Str::lower(session('chooseMood')) .'.png') }}" alt="{{ Str::lower(session('chooseMood')) }}" class='w-40 h-40'>
                </div>
            </x-slot>
            <x-slot name='body'>
                <p class='text-justify'>{{ $moodMessage[session('chooseMood')] ?? 'How are you feeling?' }}</p>
                <p class='mt-5'>Should I play something that matches your mood?</p>

                <div class="w-full flex flex-col gap-2 mt-4">

                    <form action="{{ route('preference.store') }}" method="POST" class="w-full">
                        @csrf
                        <input type="hidden" name="preference" value="match">

                        <div onclick="this.closest('form').submit()" class="cursor-pointer w-full">
                            <x-button content='Yes, Give me that' mood='{{ session("chooseMood") }}'></x-button>
                        </div>
                    </form>

                    <form action="{{ route('preference.store') }}" method="POST" class="w-full">
                        @csrf
                        <input type="hidden" name="preference" value="mismatch">

                        <div onclick="this.closest('form').submit()" class="cursor-pointer w-full">
                            <x-button content='No, Give others'></x-button>
                        </div>
                    </form>

                </div>
            </x-slot>
        </x-modal>
    @endif

    <x-modal modalId='profilePopup' additionalStyle='right-3 top-14 w-60 h-max '>
        <x-slot name='body'>
            <div class='absolute right-6 top-5' style='zoom: 0.75;' id='closeProfilePopup'>
                <x-iconButton :mood='$mood' type='cross'></x-iconButton>
            </div>
            <div class='flex flex-col items-center gap-2'>
                <div class='w-20 h-20 rounded-full flex items-center justify-center overflow-hidden {{ $bgMoodStyle[$mood] }}'>
                    @if (isset($profilePhoto))
                        <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" alt="{{ $fullname }}" class='w-full h-full object-cover'> 
                    @else
                        <p class='text-title font-primary font-bold h-fit {{ $textMoodStyle[$mood] }}'>{{ Str::charAt(Str::upper($fullname), 0) }}</p>
                    @endif
                </div>
                <div class='flex flex-col items-center'>
                    <p class='text-small font-bold {{ $textMoodStyle[$mood] }}'>{{ Str::limit($fullname, 24) }}</p>
                    <p class='text-micro text-primary-60'>{{ '@' . Str::limit($username, 18) }}</p>
                </div>
            </div>
            <hr class='my-2 border-primary-50'>
            <div class='w-full flex flex-col gap-2.5 font-secondaryAndButton text-small'>
                <a href="{{ route('user.profile') }}">
                    <div class='h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 {{ $hoverBgMoodStyle[$mood] }}'>
                        <img src="{{ asset('assets/icons/user.svg') }}" alt="user" class='w-7 h-7'>
                        <p class='text-primary-60'>Edit Your Profile</p>
                    </div>
                </a>
                <form action="{{ route('auth.logout') }}" method="POST">
                    @csrf
                    <div onclick="this.closest('form').submit()"
                        class='cursor-pointer h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 {{ $hoverBgMoodStyle[$mood] }}'>

                        <img src="{{ asset('assets/icons/logout.svg') }}" alt="logout" class='w-7 h-7'>
                        <p class='text-primary-60'>Logout</p>
                    </div>
                </form>
            </div>
        </x-slot>
    </x-modal>

    <x-modal modalId='changeMood' additionalStyle='right-20 top-14 md:right-48'>
        <x-slot name='body'>
            <p class='mb-3 font-bold text-primary-60'>Change Your Mood</p>
            <div class='w-full flex flex-col gap-2.5 font-secondaryAndButton text-small'>
                @foreach ($moodOptions as $moodOpt)
                <a href="">
                    <div class='h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 {{ (($moodOpt == $mood) ? $bgMoodStyle[$mood] . ' cursor-default disabled ' : $hoverBgMoodStyle[$mood] . ' ') }}'>
                        <img src="{{ asset('assets/moods/' . $moodOpt . '.png') }}" alt="{{ $moodOpt }}" class='w-7 h-7'>
                        <p class='text-primary-60'>{{ Str::ucfirst($moodOpt) }}</p>
                    </div>
                </a>
                @endforeach
            </div>
            <hr class='my-4'>
            <p class='mb-3 font-bold text-primary-60'>Playlist Behaviour</p>
            <div class='flex flex-row items-center gap-1.25 w-max h-max'>
                <input type="checkbox" name='playlistMood' id='playlistMood' value='1' class='w-6 h-6 rounded-lg {{ $checkboxStyle[$mood] }}'>
                <label for="playlistMood" class='text-micro md:text-small'>Based on mood</label>
            </div>
        </x-slot>
    </x-modal>
@endsection


@section('mainContent')
    <div class='flex flex-row justify-content items-center'>
        <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="" class='h-42 w-42'>
        <div class='flex flex-col text-white'>
            <p class='font-primary text-white text-title font-bold'>Hi, {{ Str::before($fullname, ' ') }}!</p>
            <p>Welcome to our life</p>
             @foreach ($playlists as $playlist)
                <p>Title: {{ $playlist->name }}</p>
            @endforeach
        </div>
    </div>
    <div>
        <p class='text-title text-secondary-relaxed-30 font-primary font-bold mt-5'>Most Current Play Songs</p>
    </div>


@endsection


@section('bottomContent')
    <x-audioPlayer :mood='$mood'></x-audioPlayer>
@endsection
