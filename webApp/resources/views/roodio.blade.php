@extends('layouts.master')


@section('title', 'ROODIO | Music Player Based on Your Mood')


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
        <x-button behaviour="navigation" navLink="login" content="Login"></x-button>
    </header>
    <section>

    </section>
    <section>

    </section>
    <footer>

    </footer>
@endsection