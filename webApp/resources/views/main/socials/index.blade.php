@extends('layouts.main')


@section('title', 'ROODIO - Socials')


@push('head')
    <meta name="csrf-token" content="{{ csrf_token() }}">
@endpush


@php
    $mainUser = auth()->user();

    $textColor = [
        'happy' => 'text-secondary-happy-30',
        'sad' => 'text-secondary-sad-30',
        'relaxed' => 'text-secondary-relaxed-30',
        'angry' => 'text-secondary-angry-30'
    ];
@endphp


@section('mainContent')
    <div class='mb-5 md:mb-9 w-full'>
        <div class='flex flex-row gap-3'>
            <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="{{ $mood }}" class='h-26 w-26 md:h-32 md:w-32 lg:h-40 lg:w-40'>
            <div class='flex flex-col'>
                <div class='w-0 relative overflow-hidden typingTextAnimation max-w-max '>
                    <p class='font-primary text-title font-bold {{ $textColor[$mood] }} md:text-hero' >Socials</p>
                </div>
                <p class='font-secondaryAndButton text-white text-justify contentFadeLoad text-small md:text-body-size'>Hi, {{ Str::before($mainUser->userDetail->fullname, ' ') }}! Let's give some social space to others.</p>
            </div>
        </div>
    </div>

    <form action="{{ route('socials.index') }}" method="GET">
        <div class='mb-7 flex flex-row gap-3 w-full lg:justify-end contentFadeLoad'>
            <x-filterButton id='all' name='filter' value='all' :mood='$mood' label='All Users' onchange="this.form.submit()"></x-filterButton>
            <x-filterButton id='following' name='filter' value='following' :mood='$mood' label='Following Only' onchange="this.form.submit()"></x-filterButton>
        </div>
    </form>

    @if (count($users) > 0)
        @if (request('filter') == 'following')
            <p class='text-white text-body-size font-secondaryAndButton w-max px-3 py-1 rounded-md '>Showing {{ $mainUser->followings()->count() . ' ' . (($mainUser->followings()->count() > 1 ? 'people' : 'person'))}} </p>
        @endif

        <div class="w-full mt-4 grid grid-cols-1 justify-items-center md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-7 md:gap-9 contentFadeLoad">
            @foreach($users as $user)
            @php
                    $username = $user->username;
                    $fullname = $user->userDetail?->fullname;
                    $profilePhoto = $user->userDetail?->profilePhoto;
                    $createdAt = floor($user->userDetail?->created_at->diffInYears(now()));
                    $followerCount = $user->followers()->count();
                    $followingCount = $user->followings()->count();
                    $checkSelfProfile = !($user->id == $mainUser->id);
                    @endphp
                @if ($checkSelfProfile)
                    <x-followCard :mood='$mood' :fullname='$fullname' :username='$username' :followingCount='$followingCount' :followerCount='$followerCount' :user='$user' :mainUser='$mainUser' :createdAt='$createdAt' :profilePhoto='$profilePhoto'></x-followCard>
                @endif
            @endforeach
        </div>
    @else
        <p class='w-full text-white font-secondaryAndButton text-small lg:text-body-size'>There is no other user(s) can be displayed.</p>
    @endif
@endsection
