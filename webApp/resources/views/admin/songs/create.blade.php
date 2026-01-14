@extends('layouts.master')

@section('title', 'Playlists Create')

@section('bodyContent')
<div class="max-w-5xl mx-auto py-10 px-4">
    <div class="mb-8">
        <h1 class="font-primary text-title text-white font-bold">Add New Song</h1>
        <p class="font-secondaryAndButton text-body-size text-shadedOfGray-20">Fill in the details below to upload a new track.</p>
    </div>

    <div class="bg-white rounded-xl shadow-lg p-8">
        <form action="{{ route('admin.songs.store') }}" method="POST" enctype="multipart/form-data">
            @csrf

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="col-span-1 md:col-span-2">
                    <label for="title" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Title</label>
                    <input type="text" name="title" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition placeholder-shadedOfGray-30" placeholder="Enter song title">
                    @error('title')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div>
                    <label for="artist" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Artist</label>
                    <input type="text" name="artist" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition" placeholder="Artist Name">
                    @error('artist')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div>
                    <label for="genre" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Genre</label>
                    <input type="text" name="genre" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition" placeholder="e.g. Pop, Jazz">
                    @error('genre')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div>
                    <label for="publisher" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Publisher</label>
                    <input type="text" name="publisher" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition" placeholder="Publisher Name">
                    @error('publisher')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div>
                    <label for="datePublished" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Date Published</label>
                    <input type="date" name="datePublished" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition text-shadedOfGray-60">
                    @error('datePublished')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div class="col-span-1 md:col-span-2">
                    <label for="lyrics" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Lyrics</label>
                    <textarea name="lyrics" rows="5" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition placeholder-shadedOfGray-30" placeholder="Type lyrics here..."></textarea>
                    @error('lyrics')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>

                <div class="col-span-1 md:col-span-2">
                    <label name="song" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Upload Song File</label>
                    <div class="flex items-center justify-center w-full">
                        <label class="flex flex-col w-full h-32 border-2 border-dashed border-shadedOfGray-30 hover:bg-shadedOfGray-10 hover:border-accent-100 rounded-lg cursor-pointer transition duration-200 relative">

                            <div class="flex flex-col items-center justify-center pt-7" id="upload-placeholder">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-shadedOfGray-40 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                </svg>
                                <span class="font-secondaryAndButton text-small text-shadedOfGray-60" id="file-name-text">Click to browse file</span>
                                <span class="font-secondaryAndButton text-micro text-shadedOfGray-40 mt-1" id="file-size-text">MP3, WAV (Max 10MB)</span>
                            </div>

                            <input type="file" name="song" id="song-input" class="opacity-0 w-full h-full absolute inset-0 cursor-pointer" accept="audio/*" />
                        </label>
                    </div>
                    @error('song')
                        <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror
                </div>
            </div>

            <div class="mt-8 flex justify-end gap-4">
                <a href="{{ route('admin.songs.index') }}" class="px-6 py-3 rounded-lg border border-shadedOfGray-30 text-shadedOfGray-60 font-secondaryAndButton text-mediumBtn hover:bg-shadedOfGray-10 transition">Cancel</a>
                <button type="submit" class="px-8 py-3 rounded-lg bg-primary-60 text-white font-secondaryAndButton text-mediumBtn font-semibold hover:bg-primary-50 shadow-lg shadow-primary-30/50 transition duration-200">Submit Song</button>
            </div>
        </form>
    </div>
</div>

<script>
    // Script untuk update nama file saat dipilih
    document.getElementById('song-input').addEventListener('change', function(e) {
        var fileName = e.target.files[0] ? e.target.files[0].name : "Click to browse file";

        // Ubah teks utama
        document.getElementById('file-name-text').textContent = fileName;
        document.getElementById('file-name-text').classList.add('text-primary-100', 'font-semibold'); // Tambah style biar terlihat beda

        // Ubah teks bantuan (opsional)
        if(e.target.files[0]) {
            document.getElementById('file-size-text').textContent = "File selected";
            document.getElementById('file-size-text').classList.add('text-accent-100');
        } else {
            document.getElementById('file-size-text').textContent = "MP3, WAV (Max 10MB)";
            document.getElementById('file-size-text').classList.remove('text-accent-100');
        }
    });
</script>
@endsection
