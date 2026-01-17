@extends('layouts.main')

@section('title', 'ROODIO - Social')

@push('head')
    <meta name="csrf-token" content="{{ csrf_token() }}">
@endpush

@section('mainContentContainerClass')

@section('mainContent')
    <form action="{{ route('social.index') }}" method="GET">

        <div>
            <input
                type="radio"
                id="all"
                name="filter"
                value="all"
                onchange="this.form.submit()"
                {{ request('filter') == 'all' || !request('filter') ? 'checked' : '' }}
            >
            <label for="all">All Users</label>
        </div>

        <div>
            <input
                type="radio"
                id="following"
                name="filter"
                value="following"
                onchange="this.form.submit()"
                {{ request('filter') == 'following' ? 'checked' : '' }}
            >
            <label for="following">Following Only</label>
        </div>

    </form>

    <div class="mt-4">
        @foreach($users as $user)
            <div class="card">
                {{ $user->username }}
            </div>
        @endforeach
    </div>
@endsection
