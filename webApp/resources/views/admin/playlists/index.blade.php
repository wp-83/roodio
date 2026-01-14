@extends('layouts.master')

@section('title', 'Playlists')

@section('bodyContent')
    <div class="font-primary bg-shadedOfGray-10 min-h-screen text-shadedOfGray-85">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
            <div class="flex justify-between items-center mb-8">
                <div>
                    <h1 class="text-title font-bold text-primary-100">My Playlists</h1>
                    <p class="text-body-size text-shadedOfGray-60 font-secondaryAndButton mt-2">
                        Koleksi musik favoritmu.
                    </p>
                </div>
                <a href="{{ route('admin.playlists.create') }}"
                class="bg-secondary-happy-100 hover:bg-secondary-happy-85 text-white font-secondaryAndButton text-mediumBtn px-6 py-3 rounded-lg shadow-md transition-colors duration-200">
                    + Buat Baru
                </a>
            </div>

            @if(session('success'))
                <div class="bg-success-lighten border border-success-moderate text-success-dark px-4 py-3 rounded mb-6 text-body-size">
                    {{ session('success') }}
                </div>
            @endif

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                @foreach($playlists as $playlist)
                <div class="bg-white rounded-xl shadow-sm border border-shadedOfGray-20 hover:shadow-lg transition-shadow duration-300 overflow-hidden flex flex-col justify-between">

                    <div class="p-6">
                        <div class="h-32 w-full bg-primary-10 rounded-lg mb-4 flex items-center justify-center bg-gradient-to-br from-primary-20 to-primary-60">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                            </svg>
                        </div>

                        <h2 class="text-subtitle font-bold text-primary-85 mb-2 truncate">
                            {{ $playlist->name }}
                        </h2>
                        <p class="text-body-size font-secondaryAndButton text-shadedOfGray-60 line-clamp-3">
                            {{ $playlist->description }}
                        </p>
                    </div>

                    <div class="bg-shadedOfGray-10 px-6 py-4 flex justify-between items-center border-t border-shadedOfGray-20">
                        <a href="{{ route('admin.playlists.edit', $playlist) }}"
                        class="text-primary-60 hover:text-primary-100 font-secondaryAndButton text-smallBtn font-semibold">
                            Edit
                        </a>

                        <form action="{{ route('admin.playlists.detsroy', $playlist) }}" method="POST" onsubmit="return confirm('Yakin ingin menghapus playlist ini?');">
                            @csrf
                            @method('DELETE')
                            <button type="submit" class="text-error-moderate hover:text-error-dark font-secondaryAndButton text-smallBtn font-semibold">
                                Delete
                            </button>
                        </form>
                    </div>
                </div>
                @endforeach
            </div>

            @if($playlists->isEmpty())
                <div class="text-center py-20">
                    <h3 class="text-subtitle text-shadedOfGray-50">Belum ada playlist.</h3>
                    <p class="text-body-size text-shadedOfGray-30">Ayo buat playlist pertamamu sekarang!</p>
                </div>
            @endif
        </div>
    </div>
@endsection
