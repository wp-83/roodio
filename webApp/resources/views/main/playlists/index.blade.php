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
    @foreach ($songs as $song)
    <!-- <h1>test</h1> -->
    <audio src="{{ config('filesystems.disks.azure.url') . '/' . $song->songPath }}" controls>asd</audio>
@endforeach

@endsection


@section('bottomContent')
    <x-audioPlayer></x-audioPlayer>
@endsection
