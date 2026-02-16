@extends('layouts.admin.master')

@section('title', 'Overview')
@section('page_title', 'Dashboard')
@section('page_subtitle', 'Welcome back, Admin.')

@section('content')
<div class="w-full space-y-8">

    {{-- 1. STATS CARDS SECTION (2 Columns) --}}
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">

        {{-- Card: Total Songs --}}
        <div class="relative overflow-hidden bg-gradient-to-br from-primary-85 to-primary-100 rounded-2xl p-6 border border-primary-70 shadow-lg group hover:border-secondary-happy-100 transition-all duration-300">
            <div class="relative z-10 flex justify-between items-start">
                <div>
                    <p class="font-secondaryAndButton text-sm text-shadedOfGray-30 font-bold uppercase tracking-wider">Total Songs</p>
                    <h3 class="font-primary text-3xl text-white font-bold mt-2">{{ number_format($totalSongs) }}</h3>
                </div>
                <div class="w-12 h-12 rounded-xl bg-primary-70/50 flex items-center justify-center text-secondary-happy-100 group-hover:scale-110 transition-transform">
                    <i class="fa-solid fa-music text-xl"></i>
                </div>
            </div>
            <div class="absolute -bottom-6 -right-6 w-24 h-24 bg-secondary-happy-100/10 rounded-full blur-2xl group-hover:bg-secondary-happy-100/20 transition-colors"></div>
        </div>

        {{-- Card: Total Playlists --}}
        <div class="relative overflow-hidden bg-gradient-to-br from-primary-85 to-primary-100 rounded-2xl p-6 border border-primary-70 shadow-lg group hover:border-accent-100 transition-all duration-300">
            <div class="relative z-10 flex justify-between items-start">
                <div>
                    <p class="font-secondaryAndButton text-sm text-shadedOfGray-30 font-bold uppercase tracking-wider">Total Playlists</p>
                    <h3 class="font-primary text-3xl text-white font-bold mt-2">{{ number_format($totalPlaylists) }}</h3>
                </div>
                <div class="w-12 h-12 rounded-xl bg-primary-70/50 flex items-center justify-center text-accent-100 group-hover:scale-110 transition-transform">
                    <i class="fa-solid fa-list text-xl"></i>
                </div>
            </div>
            <div class="absolute -bottom-6 -right-6 w-24 h-24 bg-accent-100/10 rounded-full blur-2xl group-hover:bg-accent-100/20 transition-colors"></div>
        </div>

    </div>

    {{-- 2. MAIN CONTENT GRID --}}
    <div class="grid grid-cols-1 xl:grid-cols-3 gap-8">

        {{-- LEFT COLUMN: RECENT SONGS (Wider) --}}
        <div class="xl:col-span-2">
            <div class="bg-primary-85 rounded-2xl border border-primary-70 shadow-lg overflow-hidden flex flex-col h-full">

                {{-- Header Card --}}
                <div class="p-6 border-b border-primary-70 flex justify-between items-center bg-primary-85/50 backdrop-blur-sm">
                    <h3 class="font-primary text-lg text-white font-bold">Recently Added Songs</h3>
                    <a href="{{ route('admin.songs.index') }}" class="text-xs font-bold text-primary-20 hover:text-white transition-colors">
                        VIEW ALL
                    </a>
                </div>

                {{-- Table Content --}}
                <div class="p-0 overflow-x-auto flex-grow">
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr class="text-xs text-primary-20 font-secondaryAndButton border-b border-primary-70 bg-primary-85/30">
                                <th class="px-6 py-4 font-bold uppercase tracking-wider">Title</th>
                                <th class="px-6 py-4 font-bold uppercase tracking-wider">Artist</th>
                                <th class="px-6 py-4 font-bold uppercase tracking-wider text-right">Date</th>
                            </tr>
                        </thead>
                        <tbody class="text-sm divide-y divide-primary-70">
                            @forelse($recentSongs as $song)
                                <tr class="group hover:bg-primary-70/30 transition-colors">
                                    <td class="px-6 py-4">
                                        <div class="flex items-center gap-3">
                                            <div class="w-10 h-10 rounded-lg bg-primary-70 flex items-center justify-center text-shadedOfGray-30 group-hover:text-white group-hover:bg-primary-60 transition-all">
                                                <i class="fa-solid fa-music"></i>
                                            </div>
                                            <span class="font-bold text-white">{{ $song->title }}</span>
                                        </div>
                                    </td>
                                    <td class="px-6 py-4 text-shadedOfGray-30">{{ $song->artist }}</td>
                                    <td class="px-6 py-4 text-right text-shadedOfGray-50 text-xs font-mono">
                                        {{ $song->created_at->format('d M Y') }}
                                    </td>
                                </tr>
                            @empty
                                <tr>
                                    <td colspan="3" class="px-6 py-8 text-center text-primary-20 italic">
                                        No songs uploaded yet.
                                    </td>
                                </tr>
                            @endforelse
                        </tbody>
                    </table>
                </div>

                {{-- Quick Add Song Button (PINDAH KE BAWAH) --}}
                <div class="p-4 mt-auto border-t border-primary-70">
                    <a href="{{ route('admin.songs.create') }}" class="block w-full py-3 rounded-xl border border-dashed border-secondary-happy-100 text-secondary-happy-100 text-sm font-bold text-center hover:bg-secondary-happy-100 hover:text-white transition-all">
                        + Add New Song
                    </a>
                </div>

            </div>
        </div>

        {{-- RIGHT COLUMN: RECENT PLAYLISTS --}}
        <div class="xl:col-span-1">
            <div class="bg-primary-85 rounded-2xl border border-primary-70 shadow-lg overflow-hidden flex flex-col h-full">

                {{-- Header --}}
                <div class="p-6 border-b border-primary-70 flex justify-between items-center bg-primary-85/50 backdrop-blur-sm">
                    <h3 class="font-primary text-lg text-white font-bold">New Playlists</h3>
                    <a href="{{ route('admin.playlists.index') }}" class="text-xs font-bold text-primary-20 hover:text-white transition-colors">
                        VIEW ALL
                    </a>
                </div>

                {{-- List Content --}}
                <div class="p-4 space-y-3 flex-grow">
                    @forelse($recentPlaylists as $playlist)
                        <div class="flex items-center gap-4 p-3 rounded-xl bg-primary-100/50 border border-transparent hover:border-primary-60 hover:bg-primary-70/40 transition-all group">

                            {{-- Small Cover --}}
                            <div class="relative w-12 h-12 rounded-lg bg-[#020a36] overflow-hidden flex-shrink-0">
                                @if($playlist->playlistPath)
                                    <img src="{{ config('filesystems.disks.azure.url') . '/' . $playlist->playlistPath }}"
                                         class="w-full h-full object-cover"
                                         onerror="this.onerror=null; this.parentElement.innerHTML='<i class=\'fa-solid fa-music text-white text-xs opacity-50 flex items-center justify-center h-full w-full\'></i>';">
                                @else
                                    <div class="w-full h-full flex items-center justify-center text-shadedOfGray-50">
                                        <i class="fa-solid fa-music text-xs"></i>
                                    </div>
                                @endif
                            </div>

                            {{-- Info --}}
                            <div class="flex-grow min-w-0">
                                <h4 class="text-sm font-bold text-white truncate group-hover:text-secondary-happy-100 transition-colors">
                                    {{ $playlist->name }}
                                </h4>
                                <div class="flex items-center gap-2 mt-0.5">
                                    <span class="text-[10px] text-primary-20">
                                        {{ $playlist->songs_count ?? 0 }} Songs
                                    </span>
                                </div>
                            </div>

                            {{-- Arrow Action --}}
                            <a href="{{ route('admin.playlists.edit', $playlist) }}"
                               class="w-8 h-8 rounded-full border border-primary-60 text-primary-20 flex items-center justify-center hover:bg-white hover:text-primary-100 transition-all flex-shrink-0">
                                <i class="fa-solid fa-chevron-right text-[10px]"></i>
                            </a>
                        </div>
                    @empty
                        <div class="py-8 text-center text-[#9CA3AF] italic text-sm">
                            No playlists created yet.
                        </div>
                    @endforelse
                </div>

                {{-- Quick Create Button (Footer) --}}
                <div class="p-4 mt-auto border-t border-primary-70">
                    <a href="{{ route('admin.playlists.create') }}" class="block w-full py-3 rounded-xl border border-dashed border-secondary-happy-100 text-secondary-happy-100 text-sm font-bold text-center hover:bg-secondary-happy-100 hover:text-white transition-all">
                        + Create New Playlist
                    </a>
                </div>
            </div>
        </div>

    </div>
</div>
@endsection
