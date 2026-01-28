<header class="sticky top-0 z-40 w-full px-6 py-4 md:px-8 bg-primary-100/90 backdrop-blur-xl border-b border-primary-70 transition-all duration-300">
    <div class="flex items-center justify-between">

        {{-- LEFT: Mobile Toggle & Page Title --}}
        <div class="flex items-center gap-4">

            {{-- TOMBOL MOBILE SIDEBAR (Visible only on Mobile) --}}
            <button @click="sidebarOpen = true"
                    class="md:hidden text-shadedOfGray-40 hover:text-white transition-colors p-1 focus:outline-none">
                <i class="fa-solid fa-bars text-xl"></i>
            </button>

            <div>
                <h2 class="text-lg font-bold text-white font-primary tracking-tight">@yield('page_title', 'Dashboard')</h2>
                <p class="hidden md:block text-[10px] text-shadedOfGray-40 font-secondaryAndButton uppercase tracking-wider">@yield('page_subtitle', 'Overview')</p>
            </div>
        </div>

        {{-- RIGHT: Profile Info & Logout --}}
        <div class="flex items-center gap-4 md:gap-6">

            {{-- 1. User Profile Info (Static) --}}
            <div class="flex items-center gap-3 select-none">
                <div class="text-right hidden md:block">
                    <p class="text-sm font-bold text-white font-primary leading-tight">{{ Auth::user()->fullname ?? 'Admin' }}</p>
                    <p class="text-[10px] text-shadedOfGray-40 font-secondaryAndButton group-hover:text-secondary-happy-100 transition-colors">
                        {{ '@' . (Auth::user()->username ?? 'admin') }}
                    </p>
                </div>
                <img src="https://ui-avatars.com/api/?name={{ Auth::user()->username ?? 'A' }}&background=FF8E2B&color=fff&rounded=true"
                     class="w-10 h-10 rounded-full border-2 border-primary-70 shadow-md object-cover"
                     alt="User Avatar">
            </div>

            {{-- 2. Logout Button --}}
            <form action="{{ route('auth.logout') }}" method="POST">
                @csrf
                <button type="submit"
                        class="w-10 h-10 rounded-xl border border-primary-60 bg-primary-85 text-secondary-angry-100 hover:bg-secondary-angry-100 hover:text-white transition-all flex items-center justify-center shadow-lg"
                        title="Logout">
                    <i class="bi bi-box-arrow-right text-lg"></i>
                </button>
            </form>

        </div>
    </div>
</header>
