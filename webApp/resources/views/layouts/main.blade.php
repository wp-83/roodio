@extends('layouts.master')


@section('title', 'Roodio - Get The MOO-DIES, Listen The Music')


@push('script')
    <script src="{{ asset('js/pages/navigation.js') }}" defer></script> 
@endpush


@section('bodyClass', 'max-h-screen h-screen flex flex-col')


@section('bodyContent')
    @yield('overlayContent')
    <div class="shrink-0">
        <x-navbar :mood='$mood'></x-navbar>
    </div>
    <div class='flex flex-col flex-1 min-h-0'>
        <div class='flex flex-row flex-1 min-h-0 relative'>
            <x-sidebar :mood='$mood' class='relative z-10'></x-sidebar>
            <div class='w-full bg-primary-100 overflow-x-hidden overflow-y-auto min-h-0 p-10 @yield('mainContentContainerClass')'>
                @yield('mainContent')
            </div>
        </div>
    </div>
    @yield('bottomContent')
@endsection