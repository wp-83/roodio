@extends('layouts.main')


@section('title', 'ROODIO - Moods')


@section('script')
    
@endsection


@php
    // dd($weekly, $monthly, $yearly);
    // dd($weekly[0]->mood->type);
@endphp


@section('mainContent')
    <div class='text-white'>
        {{-- @foreach ($weekly as $moodWeek)
            <p>{{ $moodWeek }}</p>
            <br><br>
        
        @endforeach --}}
        {{-- <p>{{ $weekly->id }}</p> --}}
    </div>
@endsection