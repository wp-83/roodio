@extends('layouts.master')

@section('title', 'Playlists Edit')

@section('bodyContent')
<div class="max-w-5xl mx-auto py-10 px-4">
    <div class="flex items-center justify-between mb-8">
        <div>
            <h1 class="font-primary text-title text-white font-bold">Edit Song</h1>
            <p class="font-secondaryAndButton text-body-size text-shadedOfGray-20">Update details for "{{ $song->title }}"</p>
        </div>
        <a href="{{ route('admin.songs.index') }}" class="flex items-center text-shadedOfGray-20 font-secondaryAndButton font-medium hover:text-white transition">
            <span class="mr-2">&larr;</span> Back to List
        </a>
    </div>

    <div class="bg-white rounded-xl shadow-lg p-8">
        <form action="{{ route('admin.songs.update', $song) }}" method="POST" enctype="multipart/form-data">
            @csrf
            @method('PUT')

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="col-span-1 md:col-span-2">
                    <label for="title" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Title</label>
                    <input type="text" name="title" value="{{ old('title', $song->title) }}" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition">
                    @error('title')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div>
                    <label for="artist" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Artist</label>
                    <input type="text" name="artist" value="{{ old('artist', $song->artist) }}" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition">
                    @error('artist')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div>
                    <label for="genre" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Genre</label>
                    <input type="text" name="genre" value="{{ old('genre', $song->genre) }}" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition">
                    @error('genre')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div>
                    <label for="publisher" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Publisher</label>
                    <input type="text" name="publisher" value="{{ old('publisher', $song->publisher) }}" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition">
                    @error('publisher')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div>
                    <label for="datePublished" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Date Published</label>
                    <input type="date" name="datePublished" value="{{ old('datePublished', $song->datePublished) }}" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition text-shadedOfGray-60">
                    @error('datePublished')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div class="col-span-1 md:col-span-2">
                    <label for="lyrics" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Lyrics</label>
                    <textarea name="lyrics" rows="5" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition">{{ $song->lyrics }}</textarea>
                    @error('lyrics')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>
            </div>

            <div class="mt-8 flex justify-end">
                <button type="submit" class="px-8 py-3 rounded-lg bg-primary-60 text-white font-secondaryAndButton text-mediumBtn font-semibold hover:bg-primary-50 shadow-lg shadow-primary-30/50 transition duration-200">
                    Update Song
                </button>
            </div>
        </form>
    </div>
</div>
@endsection
