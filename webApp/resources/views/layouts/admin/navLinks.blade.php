<nav class="flex-1 px-4 space-y-1 overflow-y-auto py-4">

    <p class="px-4 text-micro font-bold text-shadedOfGray-40 uppercase tracking-wider mb-2 mt-2">Menu</p>

    {{-- Overview --}}
    <a href="{{ route('admin.overview') }}"
       class="group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 {{ request()->routeIs('admin.overview') ? 'bg-primary-70 text-white shadow-lg border border-primary-60 relative overflow-hidden' : 'text-shadedOfGray-30 hover:bg-primary-70/50 hover:text-white' }}">
        @if(request()->routeIs('admin.overview'))
            <div class="absolute left-0 top-0 bottom-0 w-1 bg-secondary-happy-100 rounded-l-xl"></div>
        @endif
        <i class="bi bi-grid-1x2-fill text-lg {{ request()->routeIs('admin.overview') ? 'text-secondary-happy-100' : 'group-hover:text-white' }}"></i>
        <span class="font-medium text-small">Overview</span>
    </a>

    <p class="px-4 text-micro font-bold text-shadedOfGray-40 uppercase tracking-wider mb-2 mt-6">Library</p>

    {{-- Songs --}}
    <a href="{{ route('admin.songs.index') }}"
       class="group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 {{ request()->routeIs('admin.songs.*') ? 'bg-primary-70 text-white shadow-lg border border-primary-60 relative overflow-hidden' : 'text-shadedOfGray-30 hover:bg-primary-70/50 hover:text-white' }}">
        @if(request()->routeIs('admin.songs.*'))
            <div class="absolute left-0 top-0 bottom-0 w-1 bg-secondary-happy-100 rounded-l-xl"></div>
        @endif
        <i class="bi bi-music-note-beamed text-lg {{ request()->routeIs('admin.songs.*') ? 'text-secondary-happy-100' : 'group-hover:text-white' }}"></i>
        <span class="font-medium text-small">Songs Management</span>
    </a>

    {{-- Playlists --}}
    <a href="{{ route('admin.playlists.index') }}"
       class="group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 {{ request()->routeIs('admin.playlists.*') ? 'bg-primary-70 text-white shadow-lg border border-primary-60 relative overflow-hidden' : 'text-shadedOfGray-30 hover:bg-primary-70/50 hover:text-white' }}">
        @if(request()->routeIs('admin.playlists.*'))
            <div class="absolute left-0 top-0 bottom-0 w-1 bg-secondary-happy-100 rounded-l-xl"></div>
        @endif
        <i class="bi bi-collection-play-fill text-lg {{ request()->routeIs('admin.playlists.*') ? 'text-secondary-happy-100' : 'group-hover:text-white' }}"></i>
        <span class="font-medium text-small">Playlists</span>
    </a>
</nav>
