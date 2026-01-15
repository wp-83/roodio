@extends('layouts.main')


{{-- @section('title') --}}


@push('script')
    <script src="{{ asset('js/pages/audioControl.js') }}" defer></script>
@endpush


{{-- @section('mainContentContainerClass') --}}

@php
    $mood = 'happy';

    $moodOptions = ['happy', 'sad', 'relaxed', 'angry'];

    $moodMessage = [
        'happy' => "Ahh!! You're happy now. Stay happy and keep smiling.",
        'sad' => "Oh no… you seem sad right now. It's okay, take your time.",
        'relaxed' => 'Hmm… you look relaxed now. Enjoy the calm and breathe easy.',
        'angry' => "Whoa… you seem angry right now. Take a deep breath, it'll pass."
    ];
@endphp


@section('overlayContent')
    {{-- <form action=""></form> --}}
{{-- 
    <x-modal modalId='dayMood' additionalStyle='top-1/2 left-1/2 -translate-1/2 w-sm md:w-xl lg:w-2xl' :isNeedBg='true'>
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
    </x-modal>

    <x-modal modalId='choosePlaylist' additionalStyle='top-1/2 left-1/2 -translate-1/2 w-xs md:w-sm' :isNeedBg='true'>
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
    
    <x-modal modalId='' additionalStyle='right-0 ' :isNeedBg='false'>
        <x-slot name='header'></x-slot>
        <x-slot name='body'></x-slot>
        <x-slot name='footer'></x-slot>
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
    <x-audioPlayer></x-audioPlayer>
@endsection
