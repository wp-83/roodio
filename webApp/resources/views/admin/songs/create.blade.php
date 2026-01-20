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

                {{-- Kiri: UPLOAD FOTO COVER --}}
                <div>
                    <label class="font-bold text-gray-700 font-secondaryAndButton text-small mb-2 block">Upload Foto Cover</label>
                    <div class="w-full">
                        <label class="flex flex-col w-full aspect-square border-2 border-dashed border-shadedOfGray-30 hover:bg-shadedOfGray-10 hover:border-accent-100 rounded-lg cursor-pointer transition duration-200 relative overflow-hidden group">

                            {{-- Placeholder --}}
                            <div class="flex flex-col items-center justify-center absolute inset-0 z-10 transition-opacity duration-200 p-4 text-center" id="photo-placeholder">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-shadedOfGray-40 mb-3 group-hover:text-accent-100 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                <span class="font-secondaryAndButton text-small text-shadedOfGray-60 group-hover:text-primary-100 transition-colors">Click to upload cover</span>
                                <span class="font-secondaryAndButton text-micro text-shadedOfGray-40 mt-2">JPG, PNG (Max 2MB)</span>
                            </div>

                            {{-- Image Preview --}}
                            <img id="photo-preview" class="absolute inset-0 w-full h-full object-cover hidden z-20" alt="Cover Preview" />

                            {{-- Input File --}}
                            <input type="file" name="photo" id="photo_input" accept="image/*" class="opacity-0 w-full h-full absolute inset-0 cursor-pointer z-30" required />
                        </label>
                    </div>
                </div>

                {{-- Kanan: INPUT DETAILS (Title, Artist, Genre, Publisher, Date) --}}
                <div class="col-span-1 md:col-span-1 flex flex-col gap-5 justify-center"> {{-- Menggunakan flex-col agar semua input menumpuk di kanan --}}

                    {{-- Title --}}
                    <div>
                        <label for="title" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Title</label>
                        <input type="text" name="title" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition placeholder-shadedOfGray-30" placeholder="Enter song title">
                        @error('title')
                            <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                        @enderror
                    </div>

                    {{-- Artist --}}
                    <div>
                        <label for="artist" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Artist</label>
                        <input type="text" name="artist" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition" placeholder="Artist Name">
                        @error('artist')
                            <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                        @enderror
                    </div>

                    {{-- Genre (Dipindah ke sini) --}}
                    <div>
                        <label for="genre" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Genre</label>
                        <input type="text" name="genre" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition" placeholder="e.g. Pop, Jazz">
                        @error('genre')
                            <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                        @enderror
                    </div>

                    {{-- Publisher (Dipindah ke sini) --}}
                    <div>
                        <label for="publisher" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Publisher</label>
                        <input type="text" name="publisher" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition" placeholder="Publisher Name">
                        @error('publisher')
                            <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                        @enderror
                    </div>

                    {{-- Date Published (Dipindah ke sini) --}}
                    <div>
                        <label for="datePublished" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">Date Published</label>
                        <input type="date" name="datePublished" class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition text-shadedOfGray-60">
                        @error('datePublished')
                            <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                        @enderror
                    </div>

                </div>

                {{-- Bawah: LYRICS & FILE UPLOAD (Full Width) --}}
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
                        <label class="flex flex-col w-full h-32 border-2 border-dashed border-shadedOfGray-30 hover:bg-shadedOfGray-10 hover:border-accent-100 rounded-lg cursor-pointer transition duration-200 relative group">

                            <div class="flex flex-col items-center justify-center pt-7" id="upload-placeholder">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-shadedOfGray-40 mb-2 group-hover:text-accent-100 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                </svg>
                                <span class="font-secondaryAndButton text-small text-shadedOfGray-60 group-hover:text-primary-100 transition-colors" id="file-name-text">Click to browse file</span>
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
    // 1. Script untuk Audio Upload
    document.getElementById('song-input').addEventListener('change', function(e) {
        var fileName = e.target.files[0] ? e.target.files[0].name : "Click to browse file";
        const fileNameText = document.getElementById('file-name-text');
        fileNameText.textContent = fileName;
        fileNameText.classList.add('text-primary-100', 'font-semibold');

        const fileSizeText = document.getElementById('file-size-text');
        if(e.target.files[0]) {
            fileSizeText.textContent = "File selected";
            fileSizeText.classList.add('text-accent-100');
        } else {
            fileSizeText.textContent = "MP3, WAV (Max 10MB)";
            fileSizeText.classList.remove('text-accent-100');
        }
    });

    // 2. Script untuk Photo Upload
    document.getElementById('photo_input').addEventListener('change', function(e) {
        const file = e.target.files[0];
        const previewElement = document.getElementById('photo-preview');
        const placeholderElement = document.getElementById('photo-placeholder');

        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.src = e.target.result;
                previewElement.classList.remove('hidden');
                placeholderElement.classList.add('opacity-0');
            }
            reader.readAsDataURL(file);
        } else {
            previewElement.src = '#';
            previewElement.classList.add('hidden');
            placeholderElement.classList.remove('opacity-0');
        }
    });
</script>
@endsection
