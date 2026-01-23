<header class="sticky top-0 z-40 w-full px-6 py-4 md:px-8 bg-primary-100/90 backdrop-blur-xl border-b border-primary-70 transition-all duration-300">
    <div class="flex items-center justify-between">

        {{-- LEFT: Search Bar (Style seperti Roodio) --}}
        <div class="flex items-center flex-1 gap-4">
            {{-- Mobile Toggle --}}
            <button class="md:hidden text-white text-xl">
                <i class="bi bi-list"></i>
            </button>

            {{-- Search Input --}}
            <div class="relative w-full max-w-md hidden md:block group">
                <span class="absolute inset-y-0 left-0 flex items-center pl-4 text-shadedOfGray-30 group-focus-within:text-secondary-happy-100 transition-colors">
                    <i class="bi bi-search"></i>
                </span>
                <input type="text"
                       placeholder="Search songs, artist, lyrics..."
                       class="w-full bg-primary-85 text-white text-small font-secondaryAndButton rounded-full py-3 pl-11 pr-12 border border-primary-70 focus:outline-none focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 placeholder-shadedOfGray-40 transition-all shadow-inner">

                {{-- Shortcut Hint --}}
                <div class="absolute inset-y-0 right-0 flex items-center pr-4">
                    <span class="text-[10px] bg-primary-70 text-shadedOfGray-30 px-2 py-0.5 rounded border border-primary-60">CTRL + K</span>
                </div>
            </div>
        </div>

        {{-- RIGHT: Actions & Profile --}}
        <div class="flex items-center gap-4 md:gap-6">

            {{-- Mood/Status Badge (Static Demo) --}}
            <div class="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary-relaxed-20/10 border border-secondary-relaxed-100/30">
                <div class="w-2 h-2 rounded-full bg-secondary-relaxed-100 animate-pulse"></div>
                <span class="text-xs font-bold text-secondary-relaxed-100">System Healthy</span>
            </div>

            {{-- Notification Bell --}}
            <button class="relative w-10 h-10 flex items-center justify-center rounded-full text-shadedOfGray-20 hover:text-white hover:bg-primary-85 transition-all">
                <i class="bi bi-bell text-xl"></i>
                <span class="absolute top-2 right-2.5 w-2 h-2 bg-secondary-angry-100 rounded-full border-2 border-primary-100"></span>
            </button>

            {{-- Divider --}}
            <div class="h-8 w-px bg-primary-70 hidden md:block"></div>

            {{-- Profile Dropdown --}}
            <div class="relative" x-data="{ open: false }">
                <button onclick="document.getElementById('navProfileDropdown').classList.toggle('hidden')"
                        class="flex items-center gap-3 focus:outline-none group">
                    <div class="text-right hidden md:block">
                        <p class="text-sm font-bold text-white font-primary leading-tight">{{ Auth::user()->fullname ?? 'Admin' }}</p>
                        <p class="text-[10px] text-shadedOfGray-40 font-secondaryAndButton group-hover:text-secondary-happy-100 transition-colors">@ {{ Auth::user()->username }}</p>
                    </div>
                    <img src="https://ui-avatars.com/api/?name={{ Auth::user()->username ?? 'A' }}&background=FF8E2B&color=fff"
                         class="w-10 h-10 rounded-full border-2 border-primary-70 group-hover:border-secondary-happy-100 transition-all shadow-md">
                </button>

                {{-- Dropdown Menu --}}
                <div id="navProfileDropdown" class="hidden absolute right-0 mt-4 w-48 bg-primary-85 rounded-xl shadow-2xl border border-primary-70 py-2 z-50 transform origin-top-right transition-all">
                    <a href="#" class="flex items-center gap-3 px-4 py-2.5 text-small text-shadedOfGray-20 hover:bg-primary-70 hover:text-white transition-colors">
                        <i class="bi bi-person"></i> Profile
                    </a>
                    <a href="#" class="flex items-center gap-3 px-4 py-2.5 text-small text-shadedOfGray-20 hover:bg-primary-70 hover:text-white transition-colors">
                        <i class="bi bi-sliders"></i> Settings
                    </a>
                    <div class="border-t border-primary-70 my-1"></div>
                    <form action="{{ route('auth.logout') }}" method="POST">
                        @csrf
                        <button type="submit" class="w-full flex items-center gap-3 px-4 py-2.5 text-small text-secondary-angry-100 hover:bg-secondary-angry-20/10 transition-colors text-left">
                            <i class="bi bi-box-arrow-right"></i> Logout
                        </button>
                    </form>
                </div>
            </div>

        </div>
    </div>
</header>

<script>
    // Close dropdown when clicking outside
    document.addEventListener('click', function(event) {
        const dropdown = document.getElementById('navProfileDropdown');
        const button = event.target.closest('button');
        if (!button && !dropdown.contains(event.target)) {
            dropdown.classList.add('hidden');
        }
    });
</script>
