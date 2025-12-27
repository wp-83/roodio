<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Threads</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>
<body>
    <div class="">
        <div class="">
            <span>Title: </span>{{ $thread->title }}
            <p>{{ $thread->content }}</p>
        </div>
        <p>ini looping kumpulan reply</p>
        <div class="">
            <p>Reaction</p>
        </div>
        <div class="mt-10">
            <a href="{{route('thread.index')}}">Back</a>
        </div>
    </div>
</body>
</html>
