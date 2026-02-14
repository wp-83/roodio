@extends('layouts.main')

@section('title', 'ROODIO - Moods')

@push('style')
    <link rel="stylesheet" href="https://unpkg.com/tippy.js@6/dist/tippy.css"/>
@endpush

@push('script')
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="{{ asset('js/pages/main/mood.js') }}"></script> {{-- Removed defer to ensure window functions are available if needed, or keeping it but ensuring interaction works --}}
    <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.20/index.global.min.js'></script>
    <script src="https://unpkg.com/@popperjs/core@2"></script>
    <script src="https://unpkg.com/tippy.js@6"></script>
    <script src="https://cdn.jsdelivr.net/npm/matter-js@0.19.0/build/matter.min.js"></script>
@endpush

@section('mainContent')
    <livewire:main.moods.index />
@endsection
