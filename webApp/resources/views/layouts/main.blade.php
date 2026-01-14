@extends('layouts.master')


@section('title', 'Roodio - Get The MOO-DIES, Listen The Music')


@push('script')
    <script src="{{ asset('js/pages/navigation.js') }}" defer></script> 
@endpush


@section('bodyClass', 'max-h-screen h-screen flex flex-col')


@section('bodyContent')
    <div class="shrink-0">
        <x-navbar></x-navbar>
    </div>
    <div class='flex flex-col flex-1 min-h-0'>
        <div class='flex flex-row flex-1 min-h-0 relative'>
            <x-sidebar></x-sidebar>
            <div class='w-full bg-primary-100 overflow-x-hidden overflow-y-auto min-h-0 p-10 @yield('mainContentContainerClass')'>
                @yield('mainContent')
            </div>
        </div>
    </div>
    <x-audioPlayer></x-audioPlayer>
    {{-- <div class='w-full h-16 bg-primary-85 shrink-0'></div> --}}
@endsection