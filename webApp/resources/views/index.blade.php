@extends('layouts.main')


{{-- @section('title') --}}


@push('script')
    <script src="{{ asset('js/pages/navigation.js') }}" defer></script>
    <script src="{{ asset('js/pages/audioControl.js') }}" defer></script>   
@endpush


{{-- @section('mainContentContainerClass') --}}


@section('mainContent')
    <div clas=''>
        <p class='font-primary text-white text-title'>Hi, Andi!</p>
    </div>

    
@endsection