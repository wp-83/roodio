@extends('layouts.admin.master')

@section('title', 'Edit Song')
@section('page_title', 'Edit Song')
@section('page_subtitle', 'Update track details and media')

@section('content')
<div class="max-w-6xl mx-auto py-6">

    {{-- Header --}}
    <div class="flex items-center justify-between mb-8">
        <div>
            <h1 class="font-primary text-2xl text-white font-bold tracking-tight">Edit Song</h1>
            <p class="font-secondaryAndButton text-sm text-shadedOfGray-30 mt-1">Update details for "<span class="text-white font-semibold">{{ $song->title }}</span>"</p>
        </div>
        <a href="{{ route('admin.songs.index') }}" class="px-5 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-30 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm flex items-center gap-2">
            <i class="fa-solid fa-arrow-left"></i> Back to Library
        </a>
    </div>

    <form action="{{ route('admin.songs.update', $song) }}" method="POST" enctype="multipart/form-data">
        @csrf
        @method('PUT')

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">

            {{-- LEFT COLUMN: COVER ART & AUDIO --}}
            <div class="lg:col-span-1 space-y-6">

                {{-- Cover Art --}}
                <div class="bg-primary-85 rounded-2xl p-6 border border-primary-70 shadow-lg">
                    <label class="font-bold text-white font-primary text-sm mb-4 block">Cover Art</label>

                    {{-- Upload Box --}}
                    <div class="w-full aspect-square relative group">
                        <label class="flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-primary-60 hover:border-secondary-happy-100 hover:bg-primary-70/50 rounded-2xl cursor-pointer transition-all duration-300 overflow-hidden relative">

                            {{-- Image Preview (Show Existing or Placeholder) --}}
                            @if($song->photoPath)
                                <img id="photo-preview" src="{{ config('filesystems.disks.azure.url') . '/' . $song->photoPath }}" class="absolute inset-0 w-full h-full object-cover z-10" alt="Cover Preview" />
                            @else
                                <img id="photo-preview" class="absolute inset-0 w-full h-full object-cover hidden z-10" alt="Cover Preview" />

                                {{-- Placeholder if no image --}}
                                <div class="flex flex-col items-center justify-center text-center p-6 transition-opacity duration-300" id="photo-placeholder">
                                    <div class="w-16 h-16 rounded-full bg-primary-70 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                        <i class="fa-regular fa-image text-3xl text-shadedOfGray-40 group-hover:text-secondary-happy-100 transition-colors"></i>
                                    </div>
                                    <span class="font-secondaryAndButton text-sm text-shadedOfGray-30 font-medium group-hover:text-white transition-colors">Upload new cover</span>
                                </div>
                            @endif

                            {{-- Hover Overlay for Preview --}}
                            <div id="preview-overlay" class="absolute inset-0 bg-black/50 z-20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <span class="text-white text-sm font-bold flex items-center gap-2"><i class="fa-solid fa-pen"></i> Change Cover</span>
                            </div>

                            <input type="file" name="photo" id="photo_input" accept="image/*" class="opacity-0 w-full h-full absolute inset-0 cursor-pointer z-30" />
                        </label>
                    </div>
                    @error('photo')
                        <p class="mt-2 text-secondary-angry-100 text-xs font-medium">{{ $message }}</p>
                    @enderror
                </div>

                {{-- AUDIO FILE UPLOAD --}}
                <div class="bg-primary-85 rounded-2xl p-6 border border-primary-70 shadow-lg">
                    <label class="font-bold text-white font-primary text-sm mb-4 block">Audio File</label>

                    {{-- Current Audio Info --}}
                    @if($song->songPath)
                        <div class="mb-4 p-3 rounded-xl bg-primary-100 border border-primary-60 flex items-center gap-3">
                            <div class="w-10 h-10 rounded-full bg-secondary-happy-100/10 flex items-center justify-center text-secondary-happy-100">
                                <i class="fa-solid fa-music"></i>
                            </div>
                            <div class="flex-1 min-w-0">
                                <p class="text-xs text-shadedOfGray-30">Current File</p>
                                <p class="text-sm font-bold text-white truncate">{{ $song->songPath }}</p>
                            </div>
                        </div>
                    @endif

                    <div class="relative">
                        <label class="flex flex-col items-center justify-center w-full h-24 border-2 border-dashed border-primary-60 hover:border-accent-100 hover:bg-primary-70/50 rounded-xl cursor-pointer transition-all duration-300 group">

                            <div class="flex flex-col items-center justify-center pt-1" id="audio-placeholder">
                                <i class="fa-solid fa-cloud-arrow-up text-xl text-shadedOfGray-40 mb-1 group-hover:text-accent-100 transition-colors"></i>
                                <p class="text-xs text-shadedOfGray-30 font-medium group-hover:text-white transition-colors" id="audio-filename">Replace Audio File</p>
                            </div>

                            <input type="file" name="song" id="song-input" accept=".mp3,.wav,.aac" class="opacity-0 w-full h-full absolute inset-0 cursor-pointer" />
                        </label>
                    </div>
                    @error('song')
                        <p class="mt-2 text-secondary-angry-100 text-xs font-medium">{{ $message }}</p>
                    @enderror
                </div>
            </div>

            {{-- RIGHT COLUMN: TRACK DETAILS --}}
            <div class="lg:col-span-2">
                <div class="bg-primary-85 rounded-2xl p-8 border border-primary-70 shadow-lg h-full">
                    <h3 class="text-lg font-bold text-white mb-6 border-b border-primary-70 pb-4">Track Information</h3>

                    <div class="space-y-6">
                        {{-- Title --}}
                        <div class="space-y-2">
                            <label for="title" class="text-sm font-bold text-shadedOfGray-10">Track Title <span class="text-secondary-angry-100">*</span></label>
                            <input type="text" name="title" id="title" value="{{ old('title', $song->title) }}"
                                class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white px-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                placeholder="e.g. Bohemian Rhapsody">
                            @error('title') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                        </div>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {{-- Artist --}}
                            <div class="space-y-2">
                                <label for="artist" class="text-sm font-bold text-shadedOfGray-10">Artist Name <span class="text-secondary-angry-100">*</span></label>
                                <div class="relative">
                                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50"><i class="fa-solid fa-microphone"></i></span>
                                    <input type="text" name="artist" id="artist" value="{{ old('artist', $song->artist) }}"
                                        class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white pl-10 pr-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                        placeholder="e.g. Queen">
                                </div>
                                @error('artist') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                            </div>

                            {{-- Genre --}}
                            <div class="space-y-2">
                                <label for="genre" class="text-sm font-bold text-shadedOfGray-10">Genre <span class="text-secondary-angry-100">*</span></label>
                                <div class="relative">
                                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50"><i class="fa-solid fa-music"></i></span>
                                    <input type="text" name="genre" id="genre" value="{{ old('genre', $song->genre) }}"
                                        class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white pl-10 pr-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                        placeholder="e.g. Rock, Pop">
                                </div>
                                @error('genre') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                            </div>
                        </div>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div class="space-y-2">
                                <label for="publisher" class="text-sm font-bold text-shadedOfGray-10">Publisher <span class="text-secondary-angry-100">*</span></label>
                                <div class="relative">
                                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50"><i class="fa-solid fa-building"></i></span>
                                    <input type="text" name="publisher" id="publisher" value="{{ old('genre', $song->publisher) }}"
                                        class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white pl-10 pr-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                        placeholder="e.g. Sony Music">
                                </div>
                                @error('publisher') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                            </div>

                            <div class="space-y-2">
                                <label for="datePublished" class="text-sm font-bold text-shadedOfGray-10">Date Published <span class="text-secondary-angry-100">*</span></label>
                                <div class="relative">
                                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50"><i class="fa-regular fa-calendar"></i></span>
                                    <input type="date" name="datePublished" id="datePublished" value="{{ old('genre', $song->datePublished) }}"
                                        class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white pl-10 pr-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50 [color-scheme:dark]">
                                </div>
                                @error('datePublished') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                            </div>
                        </div>

                        {{-- Lyrics --}}
                        <div class="space-y-2">
                            <label for="lyrics" class="text-sm font-bold text-shadedOfGray-10">Lyrics <span class="text-secondary-angry-100">*</span></label>
                            <textarea name="lyrics" id="lyrics" rows="8"
                                class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white px-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50 leading-relaxed scrollbar-thin scrollbar-thumb-primary-60"
                                placeholder="Paste lyrics here...">{{ old('lyrics', $song->lyrics) }}</textarea>
                        </div>

                        {{-- Submit Buttons --}}
                        <div class="pt-6 flex items-center justify-end gap-4 border-t border-primary-70 mt-4">
                            <a href="{{ route('admin.songs.index') }}" class="px-6 py-3 rounded-xl border border-primary-60 text-shadedOfGray-30 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm">
                                Cancel
                            </a>
                            <button type="submit" class="px-8 py-3 rounded-xl bg-secondary-happy-100 text-white font-bold hover:bg-secondary-happy-85 shadow-lg shadow-secondary-happy-100/20 transition-all text-sm flex items-center gap-2">
                                <i class="fa-solid fa-floppy-disk"></i> Update Changes
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </form>
</div>

<script>
    // 1. Script untuk Audio Upload Preview Filename
    document.getElementById('song-input').addEventListener('change', function(e) {
        var fileName = e.target.files[0] ? e.target.files[0].name : "Replace Audio File";
        const fileNameText = document.getElementById('audio-filename');
        const placeholder = document.getElementById('audio-placeholder');

        fileNameText.textContent = fileName;

        if(e.target.files[0]) {
            fileNameText.classList.remove('text-shadedOfGray-30');
            fileNameText.classList.add('text-secondary-happy-100', 'font-bold');
            placeholder.querySelector('i').className = "fa-solid fa-file-audio text-xl text-secondary-happy-100 mb-1";
        } else {
            fileNameText.classList.add('text-shadedOfGray-30');
            fileNameText.classList.remove('text-secondary-happy-100', 'font-bold');
            placeholder.querySelector('i').className = "fa-solid fa-cloud-arrow-up text-xl text-shadedOfGray-40 mb-1";
        }
    });

    // 2. Script untuk Photo Upload Preview
    document.getElementById('photo_input').addEventListener('change', function(e) {
        const file = e.target.files[0];
        const previewElement = document.getElementById('photo-preview');
        const placeholderElement = document.getElementById('photo-placeholder');
        const overlayElement = document.getElementById('preview-overlay');

        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.src = e.target.result;
                previewElement.classList.remove('hidden');
                if(placeholderElement) placeholderElement.classList.add('opacity-0', 'absolute'); // Hide placeholder but keep layout
                overlayElement.classList.remove('hidden');
                overlayElement.classList.add('flex');
            }
            reader.readAsDataURL(file);
        }
    });
</script>
@endsection
