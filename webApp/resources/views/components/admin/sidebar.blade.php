{{-- ========================================== --}}
{{-- 1. MOBILE SIDEBAR (OFF-CANVAS) --}}
{{-- ========================================== --}}
<div x-show="sidebarOpen" style="display: none;" class="relative z-50 md:hidden" role="dialog" aria-modal="true">

    {{-- Backdrop (Gelap) --}}
    <div x-show="sidebarOpen"
         x-transition:enter="transition-opacity ease-linear duration-300"
         x-transition:enter-start="opacity-0"
         x-transition:enter-end="opacity-100"
         x-transition:leave="transition-opacity ease-linear duration-300"
         x-transition:leave-start="opacity-100"
         x-transition:leave-end="opacity-0"
         class="fixed inset-0 bg-primary-100/80 backdrop-blur-sm"
         @click="sidebarOpen = false"></div>

    {{-- Sidebar Panel (Slide in) --}}
    <div class="fixed inset-0 flex">
        <div x-show="sidebarOpen"
             x-transition:enter="transition ease-in-out duration-300 transform"
             x-transition:enter-start="-translate-x-full"
             x-transition:enter-end="translate-x-0"
             x-transition:leave="transition ease-in-out duration-300 transform"
             x-transition:leave-start="translate-x-0"
             x-transition:leave-end="-translate-x-full"
             class="relative mr-16 flex w-full max-w-xs flex-1">

            {{-- ISI SIDEBAR MOBILE (Sama dengan Desktop, dicopy wrapper dalamnya) --}}
            <aside class="flex flex-col w-full h-full bg-primary-85 border-r border-primary-70 shadow-2xl">
                {{-- LOGO AREA --}}
                <div class="px-8 py-8 flex items-center gap-3">
                    <div class="w-10 h-10 rounded-xl bg-white flex items-center justify-center text-primary-100 text-xl shadow-lg shadow-white/10">
                        <i class="fa-solid fa-music"></i>
                    </div>
                    <div>
                        <h1 class="font-primary font-bold text-2xl text-white tracking-tight">Roodio</h1>
                        <p class="font-secondaryAndButton text-[10px] text-shadedOfGray-30 uppercase tracking-widest font-semibold">Admin Dashboard</p>
                    </div>
                </div>

                {{-- MENU --}}
                @include('layouts.admin.navLinks') {{-- Kita pisah linknya biar rapi --}}

                {{-- USER PROFILE MOBILE --}}
                <div class="p-4 border-t border-primary-70 bg-primary-85">
                    <div class="flex items-center gap-3 p-3 rounded-xl bg-primary-70/50 border border-primary-60">
                        <img src="https://ui-avatars.com/api/?name={{ Auth::user()->username ?? 'Admin' }}&background=FF8E2B&color=fff&rounded=true"
                             class="w-10 h-10 rounded-full border-2 border-secondary-happy-100 shadow-sm" alt="Admin">
                        <div class="flex-1 min-w-0">
                            <p class="text-sm font-bold text-white truncate font-primary">{{ Auth::user()->username ?? 'Admin' }}</p>
                            <p class="text-[10px] text-secondary-happy-100 font-bold uppercase tracking-wider">Administrator</p>
                        </div>
                    </div>
                </div>
            </aside>
            {{-- END ISI SIDEBAR --}}

            {{-- Close Button (X) --}}
            <div class="absolute left-full top-0 flex w-16 justify-center pt-5">
                <button @click="sidebarOpen = false" type="button" class="-m-2.5 p-2.5 text-white hover:text-secondary-angry-100 transition-colors">
                    <span class="sr-only">Close sidebar</span>
                    <i class="fa-solid fa-xmark text-2xl"></i>
                </button>
            </div>
        </div>
    </div>
</div>


{{-- ========================================== --}}
{{-- 2. DESKTOP SIDEBAR (STATIC) --}}
{{-- ========================================== --}}
<aside class="hidden md:flex flex-col w-72 h-screen bg-primary-85 border-r border-primary-70 shadow-2xl z-30 sticky top-0">

    {{-- LOGO AREA --}}
    <div class="px-8 py-8 flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-white flex items-center justify-center text-primary-100 text-xl shadow-lg shadow-white/10">
            <i class="fa-solid fa-music"></i>
        </div>
        <div>
            <h1 class="font-primary font-bold text-2xl text-white tracking-tight">Roodio</h1>
            <p class="font-secondaryAndButton text-[10px] text-shadedOfGray-30 uppercase tracking-widest font-semibold">Admin Dashboard</p>
        </div>
    </div>

    {{-- MENU --}}
    @include('layouts.admin.navLinks')

    {{-- USER PROFILE DESKTOP --}}
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
