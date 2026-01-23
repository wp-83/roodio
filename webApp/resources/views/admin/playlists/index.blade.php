@extends('layouts.admin.master')

@section('title', 'Playlists')
@section('page_title', 'Playlists')
@section('page_subtitle', 'Manage music collections and user playlists')

@section('content')
<div class="w-full">

    {{-- HEADER SECTION --}}
    <div class="flex flex-col md:flex-row justify-between items-end mb-10 gap-6">
        <div>
            <h1 class="font-primary text-title text-white font-bold tracking-tight">Playlists Library</h1>
            <p class="font-secondaryAndButton text-body-size text-shadedOfGray-30 mt-1">Curated collections and user-generated playlists.</p>
        </div>

        {{-- Button Add New Playlist --}}
        <a href="{{ route('admin.playlists.create') }}"
           class="group bg-secondary-happy-100 hover:bg-secondary-happy-85 text-white font-secondaryAndButton font-bold px-6 py-3 rounded-xl shadow-lg shadow-secondary-happy-100/20 transition-all duration-200 transform hover:-translate-y-0.5 flex items-center gap-3 border border-secondary-happy-100/50">
            <span class="text-xl leading-none">+</span>
            <span>Create New Playlist</span>
        </a>
    </div>

    {{-- FLASH MESSAGE --}}
    @if(session('success'))
        <div id="flashMessage" class="mb-8 bg-[#0d1f67] border border-secondary-relaxed-100 text-secondary-relaxed-100 px-5 py-4 rounded-xl relative shadow-lg flex items-center gap-3 animate-fade-in-down">
            <i class="fa-solid fa-circle-check text-xl"></i>
            <span class="block sm:inline font-medium">{{ session('success') }}</span>
            <button onclick="document.getElementById('flashMessage').remove()" class="absolute top-0 bottom-0 right-0 px-4 py-3 hover:text-white transition-colors">
                <i class="fa-solid fa-xmark"></i>
            </button>
        </div>
    @endif

    {{-- PLAYLIST GRID --}}
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        @forelse($playlists as $playlist)
            <div class="group bg-primary-85 rounded-2xl border border-primary-70 shadow-lg hover:shadow-2xl hover:border-secondary-happy-100/50 transition-all duration-300 flex flex-col h-full relative overflow-hidden">

                {{-- Cover Art Area --}}
                <div class="relative w-full aspect-square bg-[#020a36] overflow-hidden">
                    {{-- Placeholder Icon (Jika tidak ada cover) --}}
                    <div class="absolute inset-0 flex items-center justify-center text-primary-60 group-hover:scale-110 transition-transform duration-500">
                        <i class="fa-solid fa-music text-6xl opacity-50"></i>
                    </div>

                    {{-- Gradient Overlay --}}
                    <div class="absolute inset-0 bg-gradient-to-t from-primary-85 via-transparent to-transparent opacity-80"></div>

                    {{-- Floating Edit Button (Muncul saat Hover) --}}
                    <div class="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <a href="{{ route('admin.playlists.edit', $playlist) }}"
                           class="w-10 h-10 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center text-white hover:bg-white hover:text-primary-100 shadow-lg transition-all border border-white/20">
                            <i class="fa-solid fa-pen text-sm"></i>
                        </a>
                    </div>
                </div>

                {{-- Content Body --}}
                <div class="p-5 flex flex-col flex-grow relative -mt-16 z-10">
                    {{-- Playlist Name --}}
                    <h3 class="font-primary text-xl font-bold text-white mb-1 line-clamp-1 group-hover:text-secondary-happy-100 transition-colors">
                        {{ $playlist->name }}
                    </h3>

                    {{-- Creator & Count --}}
                    <div class="flex items-center justify-between text-xs text-shadedOfGray-30 mb-3">
                        <span class="flex items-center gap-1.5">
                            <i class="fa-solid fa-user-circle"></i>
                            {{ $playlist->user->username ?? 'System' }}
                        </span>
                        <span class="px-2 py-0.5 rounded-md bg-primary-70 border border-primary-60 text-white font-mono">
                            {{ $playlist->songs_count ?? 0 }} Tracks
                        </span>
                    </div>

                    {{-- Description --}}
                    <p class="font-secondaryAndButton text-sm text-shadedOfGray-40 line-clamp-2 leading-relaxed mb-6 flex-grow">
                        {{ $playlist->description ?: 'No description provided.' }}
                    </p>

                    {{-- Action Footer --}}
                    <div class="pt-4 border-t border-primary-70 flex justify-between items-center">
                        <a href="{{ route('admin.playlists.edit', $playlist) }}"
                           class="text-sm font-bold text-accent-100 hover:text-white transition-colors flex items-center gap-2">
                            View Details <i class="fa-solid fa-arrow-right text-xs"></i>
                        </a>

                        <form action="{{ route('admin.playlists.destroy', $playlist) }}" method="POST" class="inline-block">
                            @csrf
                            @method('DELETE')
                            <button type="submit"
                                    onclick="return confirm('Are you sure you want to delete this playlist?')"
                                    class="text-shadedOfGray-50 hover:text-secondary-angry-100 transition-colors text-lg"
                                    title="Delete Playlist">
                                <i class="fa-regular fa-trash-can"></i>
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        @empty
            {{-- EMPTY STATE --}}
            <div class="col-span-full py-20">
                <div class="flex flex-col items-center justify-center text-center">
                    <div class="w-24 h-24 bg-primary-70/50 rounded-full flex items-center justify-center mb-6 text-shadedOfGray-40 border border-primary-60 animate-pulse">
                        <i class="fa-solid fa-list-ul text-4xl"></i>
                    </div>
                    <h3 class="font-primary text-subtitle text-white font-bold mb-2">No Playlists Found</h3>
                    <p class="font-secondaryAndButton text-shadedOfGray-30 max-w-sm mx-auto mb-8">
                        Your library is currently empty. Create a new playlist to start curating music.
                    </p>
                    <a href="{{ route('admin.playlists.create') }}" class="px-8 py-3 rounded-xl bg-primary-60 text-white font-bold hover:bg-primary-50 transition border border-primary-50 flex items-center gap-2 shadow-lg">
                        <i class="fa-solid fa-plus"></i> Create First Playlist
                    </a>
                </div>
            </div>
        @endforelse
    </div>

    {{-- PAGINATION --}}
    @if($playlists->hasPages())
        <div class="mt-12 p-4 border-t border-primary-70">
            {{ $playlists->links('pagination.superadmin') }}
        </div>
    @endif

</div>
@endsection
