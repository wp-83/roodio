@extends('layouts.master')

@section('title', 'Playlists Edit')

@section('bodyContent')
    <div class="min-h-screen flex flex-col justify-center items-center py-12 sm:px-6 lg:px-8">

    <div class="sm:mx-auto sm:w-full sm:max-w-md mb-6">
        <h2 class="text-center text-title font-bold text-accent-10">
            Edit Playlist
        </h2>
    </div>

    <div class="sm:mx-auto sm:w-full sm:max-w-md">
        <div class="bg-white py-8 px-4 shadow rounded-lg sm:px-10 border border-shadedOfGray-20">
            <form action="{{ route('admin.playlists.update', $playlist->id) }}" method="POST" class="space-y-6">
                @csrf
                @method('PUT')

                <div>
                    <label for="name" class="block text-small font-secondaryAndButton font-medium text-shadedOfGray-70">
                        Nama Playlist
                    </label>
                    <div class="mt-1">
                        <input type="text" name="name" id="name" required
                               value="{{ old('name', $playlist->name) }}"
                               class="appearance-none block w-full px-3 py-2 border border-shadedOfGray-30 rounded-md shadow-sm placeholder-shadedOfGray-30 focus:outline-none focus:ring-primary-50 focus:border-primary-50 text-body-size">
                        @error('name')
                            <p class="mt-2 text-micro text-error-moderate">{{ $message }}</p>
                        @enderror
                    </div>
                </div>

                <div>
                    <label for="description" class="block text-small font-secondaryAndButton font-medium text-shadedOfGray-70">
                        Deskripsi
                    </label>
                    <div class="mt-1">
                        <textarea id="description" name="description" rows="4"
                                  class="appearance-none block w-full px-3 py-2 border border-shadedOfGray-30 rounded-md shadow-sm placeholder-shadedOfGray-30 focus:outline-none focus:ring-primary-50 focus:border-primary-50 text-body-size">{{ old('description', $playlist->description) }}</textarea>
                        @error('description')
                            <p class="mt-2 text-micro text-error-moderate">{{ $message }}</p>
                        @enderror
                    </div>
                </div>

                <div class="flex items-center justify-end space-x-4 pt-4">
                    <a href="{{ route('admin.playlists.index') }}" class="text-shadedOfGray-60 hover:text-shadedOfGray-85 font-secondaryAndButton text-mediumBtn">
                        Batal
                    </a>
                    <button type="submit" class="flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-mediumBtn font-medium text-white bg-accent-85 hover:bg-accent-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent-50 font-secondaryAndButton">
                        Update Playlist
                    </button>
                </div>
            </form>
        </div>
    </div>
</div>
@endsection
