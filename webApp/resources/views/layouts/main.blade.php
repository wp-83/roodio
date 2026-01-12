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
        <div class='flex flex-row flex-1 min-h-0'>
            <x-sidebar></x-sidebar>
            <div class='w-full bg-primary-100 overflow-x-hidden overflow-y-auto min-h-0 p-10'>
                <p class='text-9xl text-white'>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Provident distinctio ex quasi sint debitis enim nesciunt id impedit eveniet ipsum molestias aspernatur blanditiis mollitia, voluptas saepe sequi, inventore assumenda corrupti?</p>
            </div>
        </div>
    </div>
    <div class='w-full h-16 bg-primary-85 shrink-0'></div>
@endsection