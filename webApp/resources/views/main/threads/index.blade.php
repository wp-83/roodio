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

@endphp


@section('mainContent')
    <a href="{{ route('thread.create') }}" class='group'>
        <div class='absolute z-75 right-7 bottom-7' id='threadIcon'>
            <div class="relative w-12 h-12 rounded-full border-2 bg-white {{ $borderPlusIcon[$mood] }}">
                <div class='absolute w-8 h-1 {{ $moodBaseColor[$mood] }} rounded-full top-1/2 left-1/2 -translate-1/2 md:h-1.25'></div>
                <div class='absolute w-8 h-1 {{ $moodBaseColor[$mood] }} rounded-full top-1/2 left-1/2 -translate-1/2 rotate-90 md:h-1.25'></div>
            </div>
        </div>
        <div id='threadIconLabel' class='z-10 absolute opacity-0 transition-opacity duration-150 h-max right-12 bottom-9.5 pl-2 pr-10 text-primary-70 rounded-md py-0.5 text-small {{ $bgPlusIcon[$mood] }} group-hover:opacity-100 '>
            <p>Give your opinion!</p>
        </div>
    </a>
    <div class='flex flex-col gap-30 contentFadeLoad' >
        @forelse($threads as $thread)
            <x-threadBox creator="{{ $thread->userId }}" createdAt="{{ \Carbon\Carbon::parse($thread->created_at)->diffForHumans() }}" title="{{ $thread->title }}" content="{{ $thread->content }}" :threadId='$thread->id'></x-threadBox>
        @empty
        @endforelse
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