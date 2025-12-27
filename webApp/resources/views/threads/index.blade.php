<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Threads</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>
<body>
    <a href="{{ route('thread.create') }}">Add thread</a>
    @forelse($threads as $thread)
    <div class="">
        <div class="">
            <span>Title: </span>{{ $thread->title }}
            <p>{{ $thread->content }}</p>
        </div>
        <div class="">
            <a href="{{route('thread.show', $thread->id)}}">View Detail</a>
        </div>
    </div>
    @empty
    @endforelse
</body>
</html>
