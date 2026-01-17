@extends('layouts.main')


@push('head')
    <meta name="csrf-token" content="{{ csrf_token() }}">
@endpush


@section('mainContent')
    <a href="{{ route('thread.create') }}">Add thread</a>
    @forelse($threads as $thread)
    <div class="">
        <div class="">
            <span>Title: </span>{{ $thread->title }}
            <p>{{ $thread->content }}</p>
        </div>
        <div class="">
            @session('succes')
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

            <livewire:user.reaction-button :thread-id="$thread->id" />
        </div>
    </div>
    @empty
    @endforelse

    <x-threadBox></x-threadBox>
@endsection