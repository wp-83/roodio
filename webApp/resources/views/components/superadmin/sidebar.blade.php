@php
    // Style ketika menu AKTIF
    $activeClass = 'bg-primary-60 text-white shadow-lg border-l-4 border-accent-100 font-medium';

    // Style ketika menu TIDAK AKTIF (Default)
    $inactiveClass = 'text-sm text-shadedOfGray-20 hover:bg-primary-85 hover:text-white transition-colors duration-200';

    // Base Class (Selalu dipakai keduanya)
    $baseClass = 'flex items-center gap-3 px-4 py-3 rounded-xl';
@endphp

<aside id="sidebar" class="fixed inset-y-0 left-0 z-30 w-64 bg-primary-100 text-white flex flex-col shadow-2xl transform -translate-x-full transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-auto">

    <div class="h-20 flex items-center justify-between px-6 border-b border-primary-85">
        <h1 class="font-primary text-2xl font-bold text-white tracking-wider">
            ADMIN<span class="text-accent-100">PANEL</span>
        </h1>
        <button id="close-sidebar" class="lg:hidden text-primary-30 hover:text-white">
            <i class="fa-solid fa-xmark text-xl"></i>
        </button>
    </div>

    <nav class="flex-1 overflow-y-auto py-6 px-4 space-y-2">
        <p class="px-4 text-xs text-primary-30 uppercase font-bold tracking-wider mb-2">Main Menu</p>

        {{-- 1. MENU OVERVIEW --}}
        <a href="{{ route('superadmin.users.overview') }}"
           class="{{ $baseClass }} {{ request()->routeIs('superadmin.users.overview') ? $activeClass : $inactiveClass }}">
            <i class="fa-solid fa-chart-pie w-5 text-center"></i>
            <span>Overview</span>
        </a>

        {{-- 2. MENU USER MANAGEMENT --}}
        {{-- Kita gunakan nama route spesifik 'superadmin.users.index' agar tidak tabrakan dengan overview/roles --}}
        {{-- Jika nanti ada halaman create/edit, bisa ditambahkan: request()->routeIs('superadmin.users.index', 'superadmin.users.create', 'superadmin.users.edit') --}}
        <a href="{{ route('superadmin.users.index') }}"
           class="{{ $baseClass }} {{ request()->routeIs('superadmin.users.index') ? $activeClass : $inactiveClass }}">
            <i class="fa-solid fa-users w-5 text-center"></i>
            <span>User Management</span>
        </a>

        {{-- 3. MENU ROLES --}}
        <a href="{{ route('superadmin.users.roles') }}"
           class="{{ $baseClass }} {{ request()->routeIs('superadmin.users.roles') ? $activeClass : $inactiveClass }}">
            <i class="fa-solid fa-shield-halved w-5 text-center"></i>
            <span>Roles & Access</span>
        </a>

    </nav>

    <div class="p-4 border-t border-primary-85">
        <div class="flex items-center gap-3">
            <img src="https://ui-avatars.com/api/?name=Super+Admin&background=E650C5&color=fff" alt="Admin" class="w-10 h-10 rounded-full border-2 border-accent-100">
            <div class="overflow-hidden">
                <p class="text-sm font-bold text-white truncate">{{ auth()->user()->username }}</p>
                <p class="text-xs text-primary-30 truncate">admin@system.com</p>
            </div>
        </div>
    </div>
</aside>
