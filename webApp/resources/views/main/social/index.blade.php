@extends('layouts.main')

@section('title', 'ROODIO - Social')

@push('head')
    <meta name="csrf-token" content="{{ csrf_token() }}">
@endpush

@section('mainContentContainerClass')

@section('mainContent')
<form action="{{ route('social.index') }}" method="GET">
    <div>
        <x-filterButton id='all' name='filter' value='all' :mood='$mood' label='All Users' onchange="this.form.submit()"></x-filterButton>
        <x-filterButton id='following' name='filter' value='following' :mood='$mood' label='Following Only' onchange="this.form.submit()"></x-filterButton>
    </div>
</form>

    <div class="mt-4">
        @foreach($users as $user)
            <div class="card">
                {{ $user->username }}
                <h1 class="text-white">{{ $user->followings()->count() }}</h1>
                <h1 class="text-white">{{ $user->followers()->count() }}</h1>
            </div>
        @endforeach
    </div>


@endsection
