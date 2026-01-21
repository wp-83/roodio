@extends('layouts.main')


@push('script')
    <script src="{{ asset('js/pages/main/audioControl.js') }}" defer></script>
    <script src="{{ asset('js/pages/main/index.js') }}" defer></script>
@endpush


@php
    $moodOptions = ['happy', 'sad', 'relaxed', 'angry'];

    $moodMessage = [
        'happy' => "Ahh!! You're happy now. Stay happy and keep smiling.",
        'sad' => "Oh no… you seem sad right now. It's okay, take your time.",
        'relaxed' => 'Hmm… you look relaxed now. Enjoy the calm and breathe easy.',
        'angry' => "Whoa… you seem angry right now. Take a deep breath, it'll pass."
    ];

    $bgTextName = [
        'happy' => 'bg-secondary-happy-20 text-secondary-happy-100',
        'sad' => 'bg-secondary-sad-20 text-secondary-sad-100',
        'relaxed' => 'bg-secondary-relaxed-20 text-secondary-relaxed-100',
        'angry' => 'bg-secondary-angry-20 text-secondary-angry-100'
    ];

    $contentMoodBased = [
        'happy' => 'Sounds nice! Let ROODIO increase your happiness with some melodies.',
        'sad' => 'Ahhh, Let me give you hug. ROODIO will give you special songs today.',
        'relaxed' => 'So relaxing today! Hear these songs to become more relax.',
        'angry' => 'Wow, so scary... May these songs suitable with your mood.'
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
@endsection


@section('mainContent')
    <div class='flex flex-row justify-content items-center contentFadeLoad mb-3'>
        <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="{{ $mood }}" class='h-42 w-42'>
        <div class='flex flex-col text-white'>
            <p class='font-primary text-white text-subtitle font-bold md:text-title'>Hi, <span class='{{ $bgTextName[$mood] }} pr-3 pl-1.5'>{{ Str::before($fullname, ' ') }}</span>!</p>
            <p class='font-secondaryAndButton text-small md:text-body-size'>{{ $contentMoodBased[$mood] }}</p>
        </div>
    </div>
    <div class=''>
        <div class='w-full'>
            <div class='mb-10 contentFadeLoad'>
                <div class='w-fit h-fit flex flex-row items-center gap-3 mb-3'>
                    <img src="{{ asset('assets/moods/icons/' . $mood . '.png') }}" alt="{{ $mood }}" class='w-15 h-15 md:w-24 md:h-24'>
                    <p class='text-subtitle md:text-title text-secondary-relaxed-30 font-primary font-bold'>Trending Albums</p>
                </div>
                <div class='flex flex-row'>
                    
                </div>
            </div>
            <div class='mb-10 contentFadeLoad'>
                <div class='w-fit h-fit flex flex-row items-center gap-3 mb-3'>
                    <img src="{{ asset('assets/moods/icons/' . $mood . '.png') }}" alt="{{ $mood }}" class='w-15 h-15 md:w-24 md:h-24'>
                    <p class='text-subtitle md:text-title text-secondary-relaxed-30 font-primary font-bold'>New Arrivals</p>
                </div>
                <div></div>
            </div>
            <div class='mb-5 contentFadeLoad'>
                <div class='w-fit h-fit flex flex-row items-center gap-3 mb-3'>
                    <img src="{{ asset('assets/moods/icons/' . $mood . '.png') }}" alt="{{ $mood }}" class='w-15 h-15 md:w-24 md:h-24'>
                    <p class='text-subtitle md:text-title text-secondary-relaxed-30 font-primary font-bold'>Random Mix</p>
                </div>
                <div></div>
            </div>

            {{-- <div class='flex flex-row justify-between'>
                @foreach ($playlists as $playlist)
                    <a href="/{{ $playlist->id }}">
                            <div class='w-max  overflow-hidden'>
                                <div>
                                    <div>

                                    </div>
                                    <div>

                                    </div>
                                </div>
                            </div>
                    </a>
                @endforeach
            </div> --}}
        </div>
    </div>
@endsection


@section('bottomContent')
    <x-audioPlayer :mood='$mood'></x-audioPlayer>
@endsection
