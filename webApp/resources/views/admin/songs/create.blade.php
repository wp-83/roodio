@extends('layouts.admin.master')

@section('title', 'Add New Song')
@section('page_title', 'Upload Music')
@section('page_subtitle', 'Add new tracks to the library')

@section('content')
<div class="max-w-6xl mx-auto py-6">

    {{-- Header --}}
    <div class="flex items-center justify-between mb-8">
        <div>
            <h1 class="font-primary text-2xl text-white font-bold tracking-tight">Add New Song</h1>
            <p class="font-secondaryAndButton text-sm text-shadedOfGray-30 mt-1">Fill in the details below to upload a new track.</p>
        </div>
        <a href="{{ route('admin.songs.index') }}" class="px-5 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-30 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm flex items-center gap-2">
            <i class="fa-solid fa-arrow-left"></i> Back to Library
        </a>
    </div>

    <form action="{{ route('admin.songs.store') }}" method="POST" enctype="multipart/form-data" id="upload-form">
        @csrf

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">

            {{-- LEFT COLUMN: MEDIA UPLOAD --}}
            <div class="lg:col-span-1 space-y-6">

                {{-- 1. COVER ART UPLOAD --}}
                <div class="bg-primary-85 rounded-2xl p-6 border border-primary-70 shadow-lg">
                    <label class="font-bold text-white font-primary text-sm mb-4 block">Cover Art <span class="text-secondary-angry-100">*</span></label>

                    <div class="w-full aspect-square relative group">
                        <label class="flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-primary-60 hover:border-secondary-happy-100 hover:bg-primary-70/50 rounded-2xl cursor-pointer transition-all duration-300 overflow-hidden relative">

                            {{-- Placeholder --}}
                            <div class="flex flex-col items-center justify-center text-center p-6 transition-opacity duration-300" id="photo-placeholder">
                                <div class="w-16 h-16 rounded-full bg-primary-70 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                    <i class="fa-regular fa-image text-3xl text-primary-20 group-hover:text-secondary-happy-100 transition-colors"></i>
                                </div>
                                <span class="font-secondaryAndButton text-sm text-shadedOfGray-30 font-medium group-hover:text-white transition-colors">Click to upload cover</span>
                                <span class="text-[10px] text-shadedOfGray-50 mt-2 uppercase tracking-wide">JPG, PNG (Max 5MB)</span>
                            </div>

                            {{-- Preview Image --}}
                            <img id="photo-preview" class="absolute inset-0 w-full h-full object-cover hidden z-10" alt="Cover Preview" />

                            {{-- Hover Overlay --}}
                            <div id="preview-overlay" class="absolute inset-0 bg-black/50 z-20 hidden items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <span class="text-white text-sm font-bold"><i class="fa-solid fa-pen"></i> Change Cover</span>
                            </div>

                            <input type="file" name="photo" id="photo_input" accept="image/*" class="opacity-0 w-full h-full absolute inset-0 cursor-pointer z-30" />
                        </label>
                    </div>
                    @error('photo')
                        <p class="mt-2 text-secondary-angry-100 text-xs font-medium">{{ $message }}</p>
                    @enderror
                </div>

                {{-- 2. AUDIO FILE UPLOAD --}}
                <div class="bg-primary-85 rounded-2xl p-6 border border-primary-70 shadow-lg">
                    <label class="font-bold text-white font-primary text-sm mb-4 block">Audio File <span class="text-secondary-angry-100">*</span></label>
                    <div class="relative">
                        <label class="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-primary-60 hover:border-accent-100 hover:bg-primary-70/50 rounded-xl cursor-pointer transition-all duration-300 group">

                            <div class="flex flex-col items-center justify-center pt-2" id="audio-placeholder">
                                <i class="fa-solid fa-cloud-arrow-up text-2xl text-primary-20 mb-2 group-hover:text-accent-100 transition-colors"></i>
                                <p class="text-sm text-shadedOfGray-30 font-medium group-hover:text-white transition-colors" id="audio-filename">Browse Audio File</p>
                                <p class="text-[10px] text-shadedOfGray-50 mt-1">MP3, WAV, AAC (Max 20MB)</p>
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
                            <input type="text" name="title" id="title" value="{{ old('title') }}"
                                class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white px-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                placeholder="e.g. Bohemian Rhapsody">
                            @error('title') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                        </div>

                        {{-- Artist & Genre --}}
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div class="space-y-2">
                                <label for="artist" class="text-sm font-bold text-shadedOfGray-10">Artist Name <span class="text-secondary-angry-100">*</span></label>
                                <div class="relative">
                                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50"><i class="fa-solid fa-microphone"></i></span>
                                    <input type="text" name="artist" id="artist" value="{{ old('artist') }}"
                                        class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white pl-10 pr-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                        placeholder="e.g. Queen">
                                </div>
                                @error('artist') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                            </div>

                            <div class="space-y-2">
                                <label for="genre" class="text-sm font-bold text-shadedOfGray-10">Genre <span class="text-secondary-angry-100">*</span></label>
                                <div class="relative">
                                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50"><i class="fa-solid fa-music"></i></span>
                                    <input type="text" name="genre" id="genre" value="{{ old('genre') }}"
                                        class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white pl-10 pr-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                        placeholder="e.g. Rock, Pop">
                                </div>
                                @error('genre') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                            </div>
                        </div>

                        {{-- Publisher & Date Published (NEW) --}}
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div class="space-y-2">
                                <label for="publisher" class="text-sm font-bold text-shadedOfGray-10">Publisher <span class="text-secondary-angry-100">*</span></label>
                                <div class="relative">
                                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50"><i class="fa-solid fa-building"></i></span>
                                    <input type="text" name="publisher" id="publisher" value="{{ old('publisher') }}"
                                        class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white pl-10 pr-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                        placeholder="e.g. Sony Music">
                                </div>
                                @error('publisher') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                            </div>

                            <div class="space-y-2">
                                <label for="datePublished" class="text-sm font-bold text-shadedOfGray-10">Date Published <span class="text-secondary-angry-100">*</span></label>
                                <div class="relative">
                                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50"><i class="fa-regular fa-calendar"></i></span>
                                    <input type="date" name="datePublished" id="datePublished" value="{{ old('datePublished') }}"
                                        class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white pl-10 pr-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50 [color-scheme:dark]">
                                </div>
                                @error('datePublished') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                            </div>
                        </div>

                        {{-- Lyrics --}}
                        <div class="space-y-2">
                            <label for="lyrics" class="text-sm font-bold text-shadedOfGray-10">Lyrics <span class="text-secondary-angry-100">*</span></label>
                            <textarea name="lyrics" id="lyrics" rows="6"
                                class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white px-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50 leading-relaxed scrollbar-thin scrollbar-thumb-primary-60"
                                placeholder="Paste lyrics here...">{{ old('lyrics') }}</textarea>
                            @error('lyrics') <p class="text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                        </div>

                        {{-- Submit Buttons --}}
                        <div class="pt-6 flex items-center justify-end gap-4 border-t border-primary-70 mt-4">
                            <button type="reset" class="px-6 py-3 rounded-xl border border-primary-60 text-shadedOfGray-30 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm">
                                Reset
                            </button>
                            <button type="submit" class="px-8 py-3 rounded-xl bg-secondary-happy-100 text-white font-bold hover:bg-secondary-happy-85 shadow-lg shadow-secondary-happy-100/20 transition-all text-sm flex items-center gap-2">
                                <i class="fa-solid fa-check"></i> Upload Song
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </form>
</div>

{{-- SCRIPT FOR PREVIEWS --}}
{{-- Moved to public/resources/pages/admin/upload-song.js --}}

{{-- Overlay Progress Bar --}}
{{-- Changed from fixed inset-0 to absolute inset-0 to stay within content area --}}
<div id="upload-overlay" class="absolute inset-0 bg-primary-100/90 z-50 hidden flex-col items-center justify-center backdrop-blur-sm transition-opacity duration-300 rounded-2xl">
    <div class="w-full max-w-md p-8 bg-primary-85 rounded-2xl border border-primary-70 shadow-2xl text-center relative overflow-hidden">
        {{-- Background Decoration --}}
        <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-secondary-happy-100 to-secondary-angry-100"></div>

        <div class="mb-6 relative">
            <div class="w-16 h-16 mx-auto bg-primary-70 rounded-full flex items-center justify-center mb-4">
                <i class="fa-solid fa-cloud-arrow-up text-3xl text-secondary-happy-100 animate-bounce"></i>
            </div>
            <h3 class="text-2xl font-bold text-white mb-2 font-primary" id="upload-status-title">Uploading...</h3>
            <p class="text-shadedOfGray-30 text-sm" id="upload-status-text">Please wait while we upload your track.</p>
        </div>
        
        {{-- Progress Bar --}}
        <div class="w-full bg-primary-60 rounded-full h-3 mb-4 overflow-hidden relative">
            <div id="upload-progress-bar" class="bg-gradient-to-r from-secondary-happy-100 to-accent-100 h-3 rounded-full transition-all duration-300 relative" style="width: 0%">
            </div>
        </div>
        
        <p class="text-white font-bold text-lg font-primary" id="upload-percentage">0%</p>
    </div>
</div>

{{-- Error Modal --}}
<div id="error-modal" class="absolute inset-0 bg-primary-100/90 z-[100] hidden flex-col items-center justify-center backdrop-blur-sm transition-opacity duration-300 rounded-2xl">
    <div class="w-full max-w-md p-8 bg-primary-85 rounded-2xl border border-secondary-angry-100/50 shadow-2xl text-center relative overflow-hidden animate-bounce-in">
        {{-- Background Decoration --}}
        <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-secondary-angry-100 to-red-600"></div>

        <div class="mb-6 relative">
            <div class="w-16 h-16 mx-auto bg-secondary-angry-100/10 rounded-full flex items-center justify-center mb-4 border border-secondary-angry-100/30">
                <i class="fa-solid fa-circle-exclamation text-3xl text-secondary-angry-100"></i>
            </div>
            <h3 class="text-2xl font-bold text-white mb-2 font-primary">Upload Failed</h3>
            <p class="text-shadedOfGray-30 text-sm" id="error-message-text">Something went wrong.</p>
        </div>

        <button type="button" onclick="document.getElementById('error-modal').classList.add('hidden'); document.getElementById('error-modal').classList.remove('flex');" 
                class="px-6 py-2.5 rounded-xl bg-secondary-angry-100 text-white font-bold hover:bg-secondary-angry-85 shadow-lg shadow-secondary-angry-100/20 transition-all text-sm">
            Try Again
        </button>
    </div>
</div>

@section('scripts')
    <script src="{{ asset('js/pages/admin/upload-song.js') }}"></script>
    <script>
        // Ensure globals or simple initializers if needed, but the main logic is now external
    </script>
@endsection
@endsection
