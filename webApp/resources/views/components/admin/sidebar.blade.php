<aside class="hidden md:flex flex-col w-72 h-screen bg-primary-85 border-r border-primary-70 shadow-2xl z-50">

    {{-- LOGO AREA --}}
    <div class="px-8 py-8 flex items-center gap-3">
        {{-- Logo Icon --}}
        <div class="w-10 h-10 rounded-xl bg-white flex items-center justify-center text-primary-100 text-xl shadow-lg shadow-white/10">
            <i class="fa-solid fa-music"></i>
        </div>
        <div>
            <h1 class="font-primary font-bold text-2xl text-white tracking-tight">Roodio</h1>
            <p class="font-secondaryAndButton text-[10px] text-shadedOfGray-30 uppercase tracking-widest font-semibold">Admin Dashboard</p>
        </div>
    </div>

    {{-- NAVIGATION --}}
    <nav class="flex-1 px-4 space-y-1 overflow-y-auto py-4">

        <p class="px-4 text-micro font-bold text-shadedOfGray-40 uppercase tracking-wider mb-2 mt-2">Menu</p>

        {{-- Dashboard --}}
        <a href="#" class="group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 {{ request()->routeIs('admin.dashboard') ? 'bg-primary-70 text-white shadow-lg border border-primary-60' : 'text-shadedOfGray-30 hover:bg-primary-70/50 hover:text-white' }}">
            <i class="bi bi-grid-fill text-lg {{ request()->routeIs('admin.dashboard') ? 'text-secondary-happy-100' : 'group-hover:text-white' }}"></i>
            <span class="font-medium text-small">Overview</span>
        </a>

        <p class="px-4 text-micro font-bold text-shadedOfGray-40 uppercase tracking-wider mb-2 mt-6">Library</p>

        {{-- Songs (Active State Logic) --}}
        <a href="{{ route('admin.songs.index') }}" class="group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 {{ request()->routeIs('admin.songs.*') ? 'bg-primary-70 text-white shadow-lg border border-primary-60 relative overflow-hidden' : 'text-shadedOfGray-30 hover:bg-primary-70/50 hover:text-white' }}">

            {{-- Active Indicator Line --}}
            @if(request()->routeIs('admin.songs.*'))
                <div class="absolute left-0 top-0 bottom-0 w-1 bg-secondary-happy-100 rounded-l-xl"></div>
            @endif

            <i class="bi bi-music-note-beamed text-lg {{ request()->routeIs('admin.songs.*') ? 'text-secondary-happy-100' : 'group-hover:text-white' }}"></i>
            <span class="font-medium text-small">Songs Management</span>
        </a>

        {{-- Playlists --}}
        <a href="{{ route('admin.playlists.index') }}"
           class="group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 {{ request()->routeIs('admin.playlists.*') ? 'bg-primary-70 text-white shadow-lg border border-primary-60 relative overflow-hidden' : 'text-shadedOfGray-30 hover:bg-primary-70/50 hover:text-white' }}">

            {{-- Active Indicator Line --}}
            @if(request()->routeIs('admin.playlists.*'))
                <div class="absolute left-0 top-0 bottom-0 w-1 bg-secondary-happy-100 rounded-l-xl"></div>
            @endif

            <i class="bi bi-collection-play-fill text-lg {{ request()->routeIs('admin.playlists.*') ? 'text-secondary-happy-100' : 'group-hover:text-white' }}"></i>
            <span class="font-medium text-small">Playlists</span>
        </a>

        {{-- Artists --}}
        <a href="#" class="group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 text-shadedOfGray-30 hover:bg-primary-70/50 hover:text-white">


            <i class="bi bi-people-fill text-lg group-hover:text-white"></i>
            <span class="font-medium text-small">Artists</span>
        </a>

        <p class="px-4 text-micro font-bold text-shadedOfGray-40 uppercase tracking-wider mb-2 mt-6">Settings</p>

        {{-- Settings --}}
        <a href="#" class="group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 text-shadedOfGray-30 hover:bg-primary-70/50 hover:text-white">
            <i class="bi bi-gear-fill text-lg group-hover:text-white"></i>
            <span class="font-medium text-small">Account Settings</span>
        </a>
    </nav>

    {{-- USER PROFILE (Bottom) --}}
    <div class="p-4 border-t border-primary-70 bg-primary-85">
        <div class="flex items-center gap-3 p-3 rounded-xl bg-primary-70/50 border border-primary-60">
            <img src="https://ui-avatars.com/api/?name={{ Auth::user()->username ?? 'Admin' }}&background=FF8E2B&color=fff&rounded=true"
                 class="w-10 h-10 rounded-full border-2 border-secondary-happy-100 shadow-sm" alt="Admin">
            <div class="flex-1 min-w-0">
                <p class="text-sm font-bold text-white truncate font-primary">{{ Auth::user()->username ?? 'Admin' }}</p>
                <p class="text-[10px] text-secondary-happy-100 font-bold uppercase tracking-wider">Administrator</p>
            </div>
            <form action="{{ route('auth.logout') }}" method="POST">
                @csrf
                <button type="submit" class="text-shadedOfGray-40 hover:text-white transition-colors" title="Logout">
                    <i class="bi bi-box-arrow-right text-lg"></i>
                </button>
            </form>
        </div>
    </div>
</aside>
