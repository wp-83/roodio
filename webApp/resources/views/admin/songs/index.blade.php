<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <a href="{{ route('admin.songs.create') }}">Add song</a>
    Ini adalah index admin Songs
    @forelse($songs as $song)
        <div class="">
            <p>{{ $song->id }}</p>
            <p>{{ $song->user->userDetail?->fullname }}</p>  Ini sudah benar yaa pemanggilannya tapi karena ini hasil seeder jadi emang kosong
            <p>{{ $song->title }}</p>
            <p>{{ $song->lyrics }}</p>
            <p>{{ $song->artist }}</p>
            <p>{{ $song->genre }}</p>
            <p>{{ $song->duration }}</p>
            <p>{{ $song->publisher }}</p>
            <p>{{ $song->datePublished }}</p>
            <p>{{ $song->songPath }}</p>
            <div class="">
                <a href="{{ route('admin.songs.edit', $song) }}">Update</a>
                <form action="{{ route('admin.songs.destroy', $song) }}" method="POST">
                        @csrf
                        @method('DELETE')
                        <button type="submit" onclick="return confirm('Are You Sure to delete?')">Delete</button>
                    </form>
            </div>
        </div>
    @empty
    <div class="">Yeay santai dulu lagi kosong</div>
    @endforelse

    <form action="{{ route('auth.logout') }}" method="post">
        @csrf
        <button type="submit" class="nav-link w-100 text-start">
            <i class="bi bi-door-open"></i> Logout
        </button>
    </form>
</body>
</html>
