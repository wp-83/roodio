@extends('layouts.master')


@section('title', 'ROODIO - Music Player Based on Your Mood')


@push('style')
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
@endpush


@push('script')
    <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
    <script defer>
        document.addEventListener('DOMContentLoaded', function () {
            AOS.init();
        });
    </script>
@endpush


@section('bodyContent')
    <header>
        <div class='flex flex-row items-center justify-between p-2 bg-primary-20'>
            <div class='w-64'>
                <img src="{{ asset('assets/logo/logo-horizontal.png') }}" alt="logo" class='w-full'>
            </div>
            <div class='w-max'>
                <x-button behaviour="navigation" navLink="login" content="Let's Start!"></x-button>
            </div>
        </div>
    </header>
    <section>

    </section>
    <section>

    </section>
    <footer>

    </footer>
@endsection