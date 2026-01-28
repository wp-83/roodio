<header class="h-20 bg-primary-85 border-b border-primary-70 shadow-lg flex items-center justify-between px-4 lg:px-8 z-10 shrink-0">
    <div class="flex items-center gap-4">
        {{-- Mobile Toggle --}}
        <button id="open-sidebar" class="lg:hidden text-shadedOfGray-30 p-2 -ml-2 hover:bg-primary-70 hover:text-white rounded-lg transition-colors">
            <i class="fa-solid fa-bars text-xl"></i>
        </button>

        <div>
            <h2 class="font-primary text-xl lg:text-2xl text-white font-bold tracking-tight">@yield('page_title', 'Dashboard')</h2>
            <p class="hidden sm:block text-sm text-shadedOfGray-40 font-secondaryAndButton">@yield('page_subtitle', 'Manage your application')</p>
        </div>
    </div>

    <div class="flex items-center gap-3 lg:gap-4">
        <form action="{{ route('auth.logout') }}" method="POST">
            @csrf

            {{-- Desktop Button --}}
            <div onclick="this.closest('form').submit()"
                class='cursor-pointer hidden sm:flex items-center gap-2 bg-primary-70 border border-primary-60 text-white px-5 py-2 rounded-xl text-sm font-bold hover:bg-secondary-angry-100 hover:border-secondary-angry-100 transition-all shadow-md group'>
                <span>Logout</span>
                <i class="fa-solid fa-right-from-bracket group-hover:animate-pulse"></i>
            </div>

            {{-- Mobile Button --}}
            <button type="submit" class="sm:hidden w-10 h-10 rounded-xl bg-primary-70 text-white border border-primary-60 flex items-center justify-center hover:bg-secondary-angry-100 transition-colors">
                <i class="fa-solid fa-right-from-bracket"></i>
            </button>
        </form>
    </div>
</header>
