@php
    // Style Active: Background agak terang (primary-70), text putih, border kiri oranye
    $activeClass = 'bg-primary-70 text-white shadow-lg border-l-4 border-secondary-happy-100 font-bold';

    // Style Inactive: Text abu-abu, hover jadi putih
    $inactiveClass = 'text-shadedOfGray-40 hover:bg-primary-70/50 hover:text-white transition-colors duration-200 font-medium';

    // Base Class
    $baseClass = 'flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all mb-1';
@endphp

{{-- Sidebar Container: bg-primary-85 (Dark Blue Card Color) --}}
<aside id="sidebar" class="fixed inset-y-0 left-0 z-30 w-72 bg-primary-85 border-r border-primary-70 text-white flex flex-col shadow-2xl transform -translate-x-full transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-auto">

    {{-- Logo Area --}}
    <div class="h-20 flex items-center justify-between px-6 border-b border-primary-70">
        <h1 class="font-primary text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <i class="fa-solid fa-layer-group text-secondary-happy-100"></i>
            SUPER<span class="text-secondary-happy-100">ADMIN</span>
        </h1>
        {{-- Close Button Mobile --}}
        <button id="close-sidebar" class="lg:hidden text-shadedOfGray-40 hover:text-white transition-colors">
            <i class="fa-solid fa-xmark text-xl"></i>
        </button>
    </div>

    {{-- Menu Items --}}
    <nav class="flex-1 overflow-y-auto py-6 px-4 custom-scrollbar">
        <p class="px-4 text-[10px] text-shadedOfGray-40 uppercase font-bold tracking-widest mb-3">Main Menu</p>

        {{-- 1. Overview --}}
        <a href="{{ route('superadmin.users.overview') }}"
           class="{{ $baseClass }} {{ request()->routeIs('superadmin.users.overview') ? $activeClass : $inactiveClass }}">
            <i class="fa-solid fa-chart-pie w-5 text-center {{ request()->routeIs('superadmin.users.overview') ? 'text-secondary-happy-100' : '' }}"></i>
            <span>Overview</span>
        </a>

        {{-- 2. User Management --}}
        <a href="{{ route('superadmin.users.index') }}"
           class="{{ $baseClass }} {{ request()->routeIs('superadmin.users.index') ? $activeClass : $inactiveClass }}">
            <i class="fa-solid fa-users w-5 text-center {{ request()->routeIs('superadmin.users.index') ? 'text-secondary-happy-100' : '' }}"></i>
            <span>User Management</span>
        </a>

        {{-- 3. Roles & Access --}}
        <a href="{{ route('superadmin.users.roles') }}"
           class="{{ $baseClass }} {{ request()->routeIs('superadmin.users.roles') ? $activeClass : $inactiveClass }}">
            <i class="fa-solid fa-shield-halved w-5 text-center {{ request()->routeIs('superadmin.users.roles') ? 'text-secondary-happy-100' : '' }}"></i>
            <span>Roles & Access</span>
        </a>
    </nav>

    {{-- User Profile Footer --}}
    <div class="p-4 border-t border-primary-70 bg-primary-85">
        <div class="flex items-center gap-3 p-3 rounded-xl bg-primary-70/50 border border-primary-60">
            @if (auth()->user()->userDetail?->profilePhoto)
                <img class="w-10 h-10 rounded-full object-cover border border-primary-50"
                     src="{{ config('filesystems.disks.azure.url') . '/' . auth()->user()->userDetail?->profilePhoto }}"
                     alt="{{ auth()->user()->userDetail?->fullname }}">
            @else
                <div class='w-10 h-10 rounded-full object-cover font-primary font-bold flex items-center justify-center bg-secondary-happy-100 text-white shadow-md'>
                    {{ Str::length(auth()->user()->username) > 0 ? Str::upper(Str::substr(auth()->user()->username, 0, 1)) : '?' }}
                </div>
            @endif

            <div class="overflow-hidden flex-1">
                <p class="text-sm font-bold text-white truncate">{{ auth()->user()->userDetail?->fullname ?? 'Super Admin' }}</p>
                <p class="text-[10px] text-shadedOfGray-40 truncate">{{ "@" . auth()->user()->username }}</p>
            </div>
        </div>
    </div>
</aside>
