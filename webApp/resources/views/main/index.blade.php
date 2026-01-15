@extends('layouts.main')


{{-- @section('title') --}}


@push('script')
    <script src="{{ asset('js/pages/audioControl.js') }}" defer></script>
@endpush


{{-- @section('mainContentContainerClass') --}}

@php
    $mood = 'relaxed';
@endphp


@section('mainContent')
    <div class='flex flex-row justify-content items-center'>
        <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" alt="" class='h-42 w-42'>
        <div class='flex flex-col text-white bg-accent-70'>
            <p class='font-primary text-white text-title'>Hi,
                <div>
                    {{$username}}
                </div>
            !</p>
            <p>{{ $fullname }}</p>
            <p>Welcome to our life</p>
             @foreach ($playlists as $playlist)
                <p>Title: {{ $playlist->name }}</p>
            @endforeach
        </div>
    </div>
    <div>
        <p class='text-title'>Most Current Play</p>
    </div>


@endsection


@section('bottomContent')
    <x-audioPlayer></x-audioPlayer>
@endsection
