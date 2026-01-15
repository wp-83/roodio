@extends('layouts.main')


{{-- @section('title') --}}


@push('script')
    <script src="{{ asset('js/pages/audioControl.js') }}" defer></script>
@endpush


{{-- @section('mainContentContainerClass') --}}

@php
    $mood = 'relaxed';
@endphp


@section('overlayContent')

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
