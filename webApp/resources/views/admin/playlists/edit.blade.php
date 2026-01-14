@extends('layouts.master')

@section('title', 'Playlists Edit')

@section('bodyContent')
<div class="max-w-6xl mx-auto py-10 px-4">
    <div class="flex items-center justify-between mb-8">
        <div>
            <h1 class="font-primary text-title text-white font-bold">Edit Playlist</h1>
            <p class="font-secondaryAndButton text-body-size text-shadedOfGray-20">Update detail untuk "{{ $playlist->name }}"</p>
        </div>
        <a href="{{ route('admin.playlists.index') }}" class="flex items-center text-shadedOfGray-20 font-secondaryAndButton font-medium hover:text-white transition">
            <span class="mr-2">&larr;</span> Kembali
        </a>
    </div>

    <div class="bg-white rounded-xl shadow-lg p-8">
        <form action="{{ route('admin.playlists.update', $playlist->id) }}" method="POST" enctype="multipart/form-data">
            @csrf
            @method('PUT')

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-10">

                <div class="lg:col-span-1">
                    <label class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-3">
                        Cover Playlist
                    </label>

                    <div class="relative w-full aspect-square bg-shadedOfGray-10 border-2 border-dashed border-shadedOfGray-30 rounded-xl overflow-hidden hover:bg-shadedOfGray-20 transition group cursor-pointer">

                        <input type="file" name="image" id="imageInput" accept="image/*"
                               class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                               onchange="previewImage(event)">

                        <div id="placeholder" class="absolute inset-0 flex flex-col items-center justify-center text-shadedOfGray-50 group-hover:text-primary-60 transition duration-200 {{ $playlist->playlistPath ? 'hidden' : '' }}">
                            <div class="p-4 bg-white rounded-full shadow-sm mb-3">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                            </div>
                            <span class="text-small font-secondaryAndButton font-medium">Ganti Cover</span>
                            <span class="text-micro text-shadedOfGray-40 mt-1">PNG, JPG (Max 2MB)</span>
                        </div>

                        <img id="imagePreview"
                             src="{{ config('filesystems.disks.azure.url') . '/' . $playlist->playlistPath }}"
                             alt="Preview"
                             class="absolute inset-0 w-full h-full object-cover z-10 {{ $playlist->playlistPath ? '' : 'hidden' }}">
                    </div>
                    <p class="text-micro text-shadedOfGray-40 mt-2 text-center">Biarkan kosong jika tidak ingin mengubah gambar.</p>

                    @error('image')
                        <div class="mt-2 text-error-moderate text-small font-secondaryAndButton text-center">{{ $message }}</div>
                    @enderror
                </div>

                <div class="lg:col-span-2 space-y-6">

                    <div>
                        <label for="name" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">
                            Nama Playlist
                        </label>
                        <input type="text" name="name" id="name" required
                               value="{{ old('name', $playlist->name) }}"
                               class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition">
                        @error('name')
                            <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                        @enderror
                    </div>

                    <div>
                        <label for="description" class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-2">
                            Deskripsi
                        </label>
                        <textarea id="description" name="description" rows="3"
                                  class="w-full bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size px-4 py-2 focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition">{{ old('description', $playlist->description) }}</textarea>
                        @error('description')
                            <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                        @enderror
                    </div>

                    <livewire:admin.song-picker :currentSongs="$currentSongIds" />

                    @error('songs')
                         <div class="mt-1 text-error-moderate text-small font-secondaryAndButton">{{ $message }}</div>
                    @enderror

                    <div class="pt-6 flex justify-end gap-4 border-t border-shadedOfGray-10">
                        <a href="{{ route('admin.playlists.index') }}" class="px-6 py-3 rounded-lg border border-shadedOfGray-30 text-shadedOfGray-60 font-secondaryAndButton text-mediumBtn hover:bg-shadedOfGray-10 transition">
                            Batal
                        </a>
                        <button type="submit" class="px-8 py-3 rounded-lg bg-primary-60 text-white font-secondaryAndButton text-mediumBtn font-semibold hover:bg-primary-50 shadow-lg shadow-primary-30/50 transition duration-200">
                            Update Playlist
                        </button>
                    </div>

                </div>
            </div>
        </form>
    </div>
</div>

<script>
    function previewImage(event) {
        const reader = new FileReader();
        const imageField = document.getElementById('imagePreview');
        const placeholder = document.getElementById('placeholder');

        reader.onload = function(){
            if(reader.readyState == 2){
                imageField.src = reader.result;
                imageField.classList.remove('hidden');
                placeholder.classList.add('hidden');
            }
        }

        if(event.target.files[0]) {
            reader.readAsDataURL(event.target.files[0]);
        }
    }
</script>
@endsection
