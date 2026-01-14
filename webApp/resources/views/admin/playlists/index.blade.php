@extends('layouts.master')

@section('title', 'Playlists')

@section('bodyContent')
    <div class="w-full px-6 py-10">
        <div class="max-w-7xl mx-auto">

            <div class="flex flex-col md:flex-row justify-between items-center mb-10">
                <div>
                    <h1 class="font-primary text-title text-white font-bold">My Playlists</h1>
                    <p class="font-secondaryAndButton text-body-size text-shadedOfGray-20 mt-2">
                        Koleksi musik favoritmu.
                    </p>
                </div>
                <a href="{{ route('admin.playlists.create') }}"
                class="mt-4 md:mt-0 bg-secondary-happy-100 hover:bg-secondary-happy-85 text-white font-secondaryAndButton text-mediumBtn px-6 py-3 rounded-lg shadow-md transition-colors duration-200 flex items-center gap-2">
                    <span>+ Buat Baru</span>
                </a>
            </div>

            @if(session('success'))
                <div class="bg-success-lighten border border-success-moderate text-success-dark px-4 py-3 rounded-lg mb-8 text-body-size font-secondaryAndButton shadow-sm">
                    {{ session('success') }}
                </div>
            @endif

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                @foreach($playlists as $playlist)
                <div class="bg-white rounded-xl shadow-lg border border-shadedOfGray-20 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 overflow-hidden flex flex-col h-full">

                    <div class="p-6 flex-grow">
                        <div class="h-40 w-full rounded-lg mb-6 flex items-center justify-center bg-gradient-to-br from-primary-20 to-primary-60 shadow-inner">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 text-white opacity-80" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                            </svg>
                        </div>

                        <h2 class="text-subtitle font-primary font-bold text-primary-100 mb-2 truncate">
                            {{ $playlist->name }}
                        </h2>
                        <p class="text-body-size font-secondaryAndButton text-shadedOfGray-60 line-clamp-3 leading-relaxed">
                            {{ $playlist->description }}
                        </p>
                    </div>

                    <div class="bg-shadedOfGray-10/50 px-6 py-4 flex justify-between items-center border-t border-shadedOfGray-20">
                        <a href="{{ route('admin.playlists.edit', $playlist) }}"
                        class="text-primary-60 hover:text-primary-100 font-secondaryAndButton text-smallBtn font-semibold transition-colors">
                            Edit Playlist
                        </a>

                        <form action="{{ route('admin.playlists.destroy', $playlist) }}" method="POST" onsubmit="return confirm('Yakin ingin menghapus playlist ini?');">
                            @csrf
                            @method('DELETE')
                            <button type="submit" class="text-error-moderate hover:text-error-dark font-secondaryAndButton text-smallBtn font-semibold transition-colors">
                                Delete
                            </button>
                        </form>
                    </div>
                </div>
                @endforeach
            </div>

            @if($playlists->isEmpty())
                <div class="text-center py-20 bg-white/5 rounded-xl border border-white/10 mt-8">
                    <h3 class="font-primary text-subtitle text-white mb-2">Belum ada playlist.</h3>
                    <p class="font-secondaryAndButton text-body-size text-shadedOfGray-20">Ayo buat playlist pertamamu sekarang!</p>
                </div>
            @endif
        </div>
    </div>
@endsection
