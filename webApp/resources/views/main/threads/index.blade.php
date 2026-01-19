@extends('layouts.main')


@section('title', 'ROODIO - Threads')


@push('head')
    <meta name="csrf-token" content="{{ csrf_token() }}">
@endpush


@push('script')
    <script src="{{ asset('js/pages/main/thread.js') }}" defer></script>
@endpush


@section('mainContentContainerClass')

@php
    $bgPlusIcon = [
        'happy' => 'bg-secondary-happy-10',
        'sad' => 'bg-secondary-sad-10',
        'relaxed' => 'bg-secondary-relaxed-10',
        'angry' => 'bg-secondary-angry-10'
    ];

    $borderPlusIcon = [
        'happy' => 'border-secondary-happy-85',
        'sad' => 'border-secondary-sad-85',
        'relaxed' => 'border-secondary-relaxed-85',
        'angry' => 'border-secondary-angry-85'
    ];

    $moodBaseColor = [
        'happy' => 'bg-secondary-happy-85',
        'sad' => 'bg-secondary-sad-85',
        'relaxed' => 'bg-secondary-relaxed-85',
        'angry' => 'bg-secondary-angry-85'
    ];

    $textColor = [
        'happy' => 'text-secondary-happy-30',
        'sad' => 'text-secondary-sad-30',
        'relaxed' => 'text-secondary-relaxed-30',
        'angry' => 'text-secondary-angry-30'
    ];

@endphp


@section('mainContent')
    <div class='mb-6 md:mb-11'>
        <div class='flex w-full flex-row justify-between items-center'>
            <div class='flex flex-col'>
                <div class='w-0 relative overflow-hidden typingTextAnimation max-w-max '>
                    <p class='font-primary text-title font-bold {{ $textColor[$mood] }} md:text-hero' >Threads</p>
                </div>
                <p class='font-secondaryAndButton text-white text-justify contentFadeLoad text-small md:text-body-size'>Nice to meet you again, {{ Str::before($fullname, ' ') }}! Go ahead and express how you feel.</p>
            </div>
            <div class='w-max invisible lg:visible contentFadeLoad'>
                <a href="{{ route('thread.create') }}">
                    <x-button content='Add new thread' :mood='$mood'></x-button>
                </a>
            </div>
        </div>
    </div>
    <a href="{{ route('thread.create') }}" class='group'>
        <div class='absolute z-75 right-7 bottom-7' id='threadIcon'>
            <div class="relative w-12 h-12 rounded-full border-2 bg-white {{ $borderPlusIcon[$mood] }}">
                <div class='absolute w-8 h-1 {{ $moodBaseColor[$mood] }} rounded-full top-1/2 left-1/2 -translate-1/2 md:h-1.25'></div>
                <div class='absolute w-8 h-1 {{ $moodBaseColor[$mood] }} rounded-full top-1/2 left-1/2 -translate-1/2 rotate-90 md:h-1.25'></div>
            </div>
        </div>
        <div id='threadIconLabel' class='z-10 absolute opacity-0 transition-opacity duration-150 h-max right-12 bottom-9.5 pl-2 pr-10 text-primary-70 rounded-md py-0.5 text-small border-2 {{ $bgPlusIcon[$mood] . ' ' . $borderPlusIcon[$mood] }} group-hover:opacity-100 '>
            <p>Give your opinion!</p>
        </div>
    </a>
    <div class='flex flex-col gap-8 contentFadeLoad' >
        @forelse($threads as $thread)
            
            <x-threadBox mood='{{ $mood }}' creator="{{ $thread->user->userDetail->fullname }}" profilePicture='{{ $thread->user->userDetail->profilePhoto }}' createdAt="{{ \Carbon\Carbon::parse($thread->created_at)->diffForHumans() }}" title="{{ $thread->title }}" content="{{ $thread->content }}" :threadId='$thread->id' :isReplyable='$thread->isReplyable'></x-threadBox>
        @empty
        @endforelse
    </div>
    <div class='w-full flex justify-start'>
        {{ $threads->links() }}
    </div>
@endsection

{{--
    <div class="">
        <div class="">
            <span>Title: </span>{{ $thread->title }}
            <p>{{ $thread->content }}</p>
        </div>
        <div class="">
            @session('success')
            <strong>Success! {{ $value }}</strong>
            @endsession
            @forelse($thread->replies as $reply)
                <p>{{ $reply->content }}</p>
            @empty
            @endforelse
            <div class="">
                <form action="{{ route('thread.reply', $thread->id) }}" method="POST">
                    @csrf
                    <label for="content">Reply:</label>
                    <textarea name="content" class="border"></textarea>
                    <button type="submit">send</button>
                </form>
                @error('content')
                    {{ $message }}
                @enderror
            </div>
        </div>  --}}
