<header class="h-20 bg-white shadow-sm flex items-center justify-between px-4 lg:px-8 z-10 shrink-0">
    <div class="flex items-center gap-4">
        <button id="open-sidebar" class="lg:hidden text-primary-100 p-2 -ml-2 hover:bg-primary-10 rounded-lg">
            <i class="fa-solid fa-bars text-xl"></i>
        </button>

        <div>
            <h2 class="font-primary text-xl lg:text-2xl text-primary-100 font-bold">@yield('page_title', 'Dashboard')</h2>
            <p class="hidden sm:block text-sm text-shadedOfGray-60">@yield('page_subtitle', 'Manage your application')</p>
        </div>
    </div>

    <div class="flex items-center gap-3 lg:gap-4">
        <button class="w-10 h-10 rounded-full bg-shadedOfGray-10 text-primary-85 hover:bg-primary-10 hover:text-primary-100 transition-colors relative">
            <i class="fa-regular fa-bell"></i>
            <span class="absolute top-2 right-2 w-2.5 h-2.5 bg-accent-100 rounded-full border-2 border-white"></span>
        </button>
        <form action="{{ route('auth.logout') }}" method="POST">
            @csrf
            <div onclick="this.closest('form').submit()"
                class='cursor-pointer hidden sm:block bg-primary-100 text-white px-5 py-2 rounded-lg text-sm font-medium hover:bg-primary-85 transition-colors shadow-lg shadow-primary-30/40'>
                Logout
            </div>
            <button type="submit" class="sm:hidden w-10 h-10 rounded-lg bg-primary-100 text-white flex items-center justify-center">
                <i class="fa-solid fa-right-from-bracket"></i>
            </button>
        </form>
    </div>
</header>
