<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <title>@yield('title') | Roodio Admin</title>

    {{-- Fonts --}}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Urbanist:wght@400;500;600;700&display=swap" rel="stylesheet">

    {{-- Icons --}}
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    {{-- Icons --}}
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    {{-- CSS & JS Assets --}}
    @vite(['resources/css/app.css', 'resources/js/app.js'])
    @livewireStyles
</head>

{{-- TAMBAHKAN x-data DISINI UNTUK MENGONTROL SIDEBAR --}}
<body class="bg-primary-100 text-white antialiased selection:bg-secondary-happy-100 selection:text-white"
      x-data="{ sidebarOpen: false }">

    {{-- Main Container (Hapus overflow-hidden di body, pindah ke sini) --}}
    <div class="flex h-screen w-full overflow-hidden">

        {{--
            INCLUDE SIDEBAR (Responsive)
            Pastikan file ini berisi kode sidebar responsive yang saya berikan sebelumnya
        --}}
        @include('components.admin.sidebar')

        {{-- CONTENT WRAPPER --}}
        <div class="relative flex flex-col flex-1 overflow-y-auto overflow-x-hidden">

            {{--
                INCLUDE HEADER / NAVBAR (Responsive)
                Pastikan file ini berisi kode header dengan tombol hamburger menu
            --}}
            @include('components.admin.navbar')

            {{-- CONTENT SCROLL AREA --}}
            <main class="flex-1 p-6 md:p-8 scroll-smooth relative">

                {{-- Background Gradient Decoration --}}
                <div class="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-primary-85/50 to-transparent pointer-events-none z-0"></div>

                {{-- Actual Content --}}
                <div class="relative z-10 min-h-[80vh]">
                    @yield('content')
                </div>

                {{-- Footer --}}
                <div class="mt-12 py-6 border-t border-primary-70 text-center md:text-right text-primary-20 text-micro relative z-10">
                    <p>&copy; {{ date('Y') }} Roodio Music. Admin Panel.</p>
                </div>
            </main>
        </div>
    </div>

    @livewireScripts
</body>
</html>
