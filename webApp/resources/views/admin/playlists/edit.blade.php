@extends('layouts.admin.master')

@section('title', 'Edit Playlist')
@section('page_title', 'Edit Playlist')
@section('page_subtitle', 'Update playlist details and tracks')

@section('content')
<div class="max-w-7xl mx-auto py-6" x-data="{ showSongModal: false }">

    {{-- Header --}}
    <div class="flex items-center justify-between mb-8">
        <div>
            <h1 class="font-primary text-2xl text-white font-bold tracking-tight">Edit Playlist</h1>
            <p class="font-secondaryAndButton text-sm text-shadedOfGray-30 mt-1">Update details for "<span class="text-white font-bold">{{ $playlist->name }}</span>".</p>
        </div>
        <a href="{{ route('admin.playlists.index') }}" class="px-5 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-30 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm flex items-center gap-2">
            <i class="fa-solid fa-arrow-left"></i> Back to Library
        </a>
    </div>

    <form action="{{ route('admin.playlists.update', $playlist->id) }}" method="POST" enctype="multipart/form-data">
        @csrf
        @method('PUT')

        <div class="grid grid-cols-1 xl:grid-cols-12 gap-8">

            {{-- COLUMN 1: PLAYLIST DETAILS (Lebar 4/12) --}}
            <div class="xl:col-span-4 space-y-6">

                {{-- Card: Basic Info --}}
                <div class="bg-primary-85 rounded-2xl p-6 border border-primary-70 shadow-lg sticky top-6">
                    <h3 class="text-lg font-bold text-white mb-6 border-b border-primary-70 pb-4">Playlist Details</h3>

                    {{-- Cover Image Upload --}}
                    <div class="mb-6">
                        <label class="font-bold text-white font-primary text-sm mb-3 block">Cover Image</label>
                        <div class="w-full aspect-square relative group">
                            <label class="flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-primary-60 hover:border-secondary-happy-100 hover:bg-primary-70/50 rounded-2xl cursor-pointer transition-all duration-300 overflow-hidden relative">

                                {{-- Placeholder (Hanya muncul jika tidak ada gambar lama & baru) --}}
                                <div class="flex flex-col items-center justify-center text-center p-6 {{ $playlist->playlistPath ? 'hidden' : '' }}" id="photo-placeholder">
                                    <div class="w-14 h-14 rounded-full bg-primary-70 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                                        <i class="fa-regular fa-image text-2xl text-[#9CA3AF] group-hover:text-secondary-happy-100 transition-colors"></i>
                                    </div>
                                    <span class="text-xs text-shadedOfGray-30 font-medium">Change Cover</span>
                                </div>

                                {{-- Image Preview (Menampilkan gambar lama dari Azure atau preview upload baru) --}}
                                <img id="imagePreview"
                                     src="{{ $playlist->playlistPath ? config('filesystems.disks.azure.url') . '/' . $playlist->playlistPath : '#' }}"
                                     alt="Preview"
                                     class="absolute inset-0 w-full h-full object-cover z-10 {{ $playlist->playlistPath ? '' : 'hidden' }}" />

                                {{-- Overlay saat hover --}}
                                <div class="absolute inset-0 bg-black/50 z-20 hidden group-hover:flex items-center justify-center transition-opacity">
                                    <span class="text-white text-xs font-bold"><i class="fa-solid fa-pen"></i> Change</span>
                                </div>

                                <input type="file" name="image" id="imageInput" accept="image/png,jpg,jpeg" class="hidden" onchange="previewImage(event)" />
                            </label>
                        </div>
                        @error('image') <p class="mt-2 text-secondary-angry-100 text-xs">{{ $message }}</p> @enderror
                    </div>

                    {{-- Name --}}
                    <div class="space-y-4">
                        <div>
                            <label class="text-sm font-bold text-shadedOfGray-10 mb-2 block">Name <span class="text-secondary-angry-100">*</span></label>
                            <input type="text" name="name" value="{{ old('name', $playlist->name) }}"
                                class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white px-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50"
                                placeholder="e.g. Late Night Lo-Fi" required>
                            @error('name') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        {{-- Description --}}
                        <div>
                            <label class="text-sm font-bold text-shadedOfGray-10 mb-2 block">Description</label>
                            <textarea name="description" rows="4"
                                class="w-full bg-primary-100 border border-primary-60 rounded-xl text-white px-4 py-3 focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 outline-none transition placeholder-shadedOfGray-50 leading-relaxed resize-none"
                                placeholder="What is this playlist about?">{{ old('description', $playlist->description) }}</textarea>
                        </div>
                    </div>

                    {{-- Submit Button --}}
                    <div class="mt-8 pt-6 border-t border-primary-70 flex gap-3">
                        <a href="{{ route('admin.playlists.index') }}" class="flex-1 py-3.5 rounded-xl border border-primary-60 text-shadedOfGray-30 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm text-center">
                            Cancel
                        </a>
                        <button type="submit" class="flex-[2] py-3.5 rounded-xl bg-secondary-happy-100 text-white font-bold hover:bg-secondary-happy-85 shadow-lg shadow-secondary-happy-100/20 transition-all text-sm flex items-center justify-center gap-2">
                            <i class="fa-solid fa-save"></i> Update Playlist
                        </button>
                    </div>
                </div>
            </div>

            {{-- COLUMN 2: SONG MANAGEMENT (Lebar 8/12) --}}
            <div class="xl:col-span-8">
                <div class="bg-primary-85 rounded-2xl border border-primary-70 shadow-lg flex flex-col h-full min-h-[600px]">

                    {{-- Header Area --}}
                    <div class="p-6 border-b border-primary-70 flex justify-between items-center">
                        <div>
                            <h3 class="text-lg font-bold text-white">Tracks</h3>
                            <p class="text-xs text-shadedOfGray-30 mt-1">Manage songs included in this playlist.</p>
                        </div>

                        {{-- Trigger Modal Button --}}
                        <button type="button" @click="showSongModal = true"
                            class="px-5 py-2.5 rounded-xl bg-primary-100 border border-secondary-happy-100 text-secondary-happy-100 font-bold hover:bg-secondary-happy-100 hover:text-white transition-all text-sm flex items-center gap-2 shadow-md">
                            <i class="fa-solid fa-plus"></i> Add Tracks
                        </button>
                    </div>

                    {{-- Area List Lagu Terpilih --}}
                    <div class="p-6 flex-grow">
                        {{--
                            PENTING:
                            Kita passing data lagu yang sudah ada ($playlist->songs) ke component Livewire
                            agar listnya tidak kosong saat halaman Edit dibuka.

                            Pastikan di PlaylistSelectedSongs.php ada method mount($initialSongs = [])
                        --}}
                        <livewire:admin.playlist-selected-songs :initialSongs="$playlist->songs" />

                        @error('songs')
                            <div class="mt-4 p-3 rounded-lg bg-secondary-angry-100/10 border border-secondary-angry-100/20 text-secondary-angry-100 text-sm flex items-center gap-2">
                                <i class="fa-solid fa-circle-exclamation"></i> {{ $message }}
                            </div>
                        @enderror
                    </div>
                </div>
            </div>
        </div>

        {{-- ================================================= --}}
        {{-- MODAL: SONG PICKER (ADD SONGS)                    --}}
        {{-- ================================================= --}}
        <template x-teleport="body">
            <div x-show="showSongModal" style="display: none;"
                 class="fixed inset-0 z-[9999] overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">

                {{-- Backdrop --}}
                <div x-show="showSongModal"
                     x-transition:enter="ease-out duration-300" x-transition:enter-start="opacity-0" x-transition:enter-end="opacity-100"
                     x-transition:leave="ease-in duration-200" x-transition:leave-start="opacity-100" x-transition:leave-end="opacity-0"
                     class="fixed inset-0 bg-[#020a36]/90 backdrop-blur-sm transition-opacity"
                     @click="showSongModal = false"></div>

                {{-- Modal Panel --}}
                <div class="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
                    <span class="hidden sm:inline-block sm:align-middle sm:h-screen">&#8203;</span>

                    <div x-show="showSongModal"
                         x-transition:enter="ease-out duration-300" x-transition:enter-start="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95" x-transition:enter-end="opacity-100 translate-y-0 sm:scale-100"
                         x-transition:leave="ease-in duration-200" x-transition:leave-start="opacity-100 translate-y-0 sm:scale-100" x-transition:leave-end="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                         class="relative inline-block align-bottom bg-primary-85 rounded-2xl text-left overflow-hidden shadow-2xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl w-full border border-primary-70">

                        {{-- Modal Header --}}
                        <div class="px-6 py-4 border-b border-primary-70 flex justify-between items-center bg-primary-85 sticky top-0 z-10">
                            <h3 class="text-lg leading-6 font-bold text-white font-primary">Browse Library</h3>
                            <button type="button" @click="showSongModal = false" class="text-[#9CA3AF] hover:text-white transition-colors">
                                <i class="fa-solid fa-xmark text-xl"></i>
                            </button>
                        </div>

                        {{-- Modal Body: LIVEWIRE SONG PICKER --}}
                        {{-- Kita passing juga array ID lagu yang sudah ada agar tombolnya "Added" (Disabled) --}}
                        <div class="bg-primary-100 p-6 max-h-[70vh] overflow-y-auto custom-scrollbar">
                            <livewire:admin.song-picker :currentSongIds="$playlist->songs->pluck('id')->toArray()" />
                        </div>

                        {{-- Modal Footer --}}
                        <div class="px-6 py-4 border-t border-primary-70 bg-primary-85 flex justify-end gap-3">
                            <button type="button" @click="showSongModal = false" class="px-6 py-2.5 rounded-xl bg-secondary-happy-100 text-white font-bold hover:bg-secondary-happy-85 transition-colors text-sm shadow-lg">
                                Done Selecting
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </template>

    </form>
</div>

{{-- SCRIPT: Image Preview --}}
<script>
    function previewImage(event) {
        const reader = new FileReader();
        const imageField = document.getElementById('imagePreview');
        const placeholder = document.getElementById('photo-placeholder');

        reader.onload = function(){
            if(reader.readyState == 2){
                imageField.src = reader.result;
                imageField.classList.remove('hidden');
                if(placeholder) placeholder.classList.add('hidden');
            }
        }
        if(event.target.files[0]) {
            reader.readAsDataURL(event.target.files[0]);
        }
    }
</script>
@endsection
