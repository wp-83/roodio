<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edit Song</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>
<body>
    <a href="{{ route('admin.songs.index') }}">Back</a>
    <form action="{{ route('admin.songs.update', $song) }}" method="POST" enctype="multipart/form-data">
        @csrf
        @method('PUT')
        <label for="title">Title</label>
        <input type="text" name="title" value="{{ old('title', $song->title) }}">
        @error('title')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="lyrics">Lyrics</label>
        <textarea name="lyrics">{{ $song->lyrics }}</textarea>
        @error('lyrics')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="artist">Artist</label>
        <input type="text" name="artist" value="{{ old('artist', $song->artist) }}">
        @error('artist')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="genre">genre</label>
        <input type="text" name="genre" value="{{ old('genre', $song->genre) }}">
        @error('genre')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="publisher">publisher</label>
        <input type="text" name="publisher" value="{{ old('publisher', $song->publisher) }}">
        @error('publisher')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="datePublished"></label>
        <input type="date" name="datePublished" value="{{ old('datePublished', $song->datePublished) }}">
        @error('datePublished')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <button type="submit">Submit</button>
    </form>
</body>
</html>
