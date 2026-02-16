<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>@yield('title', 'Super Admin Panel')</title>

    {{-- Fonts --}}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Urbanist:wght@400;500;600;700&display=swap" rel="stylesheet">

    {{-- Icons --}}
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: {
                            100: '#020A36', // Background Utama Deep Blue
                            85: '#06134D',  // Card / Sidebar / Header
                            70: '#0D1F67',  // Border
                            60: '#142C80',
                            50: '#1F3A98',
                            30: '#4F6CC3',
                            20: '#7591DB',
                            10: '#A4BEF2',
                        },
                        secondary: {
                            happy: { 100: '#FF8E2B', 85: '#FFA350', 20: '#FFF2E5' },
                            sad: { 100: '#6A4FBF', 20: '#EEE8FB' },
                            relaxed: { 100: '#28C76F', 20: '#E0F7EB' },
                            angry: { 100: '#E63946', 20: '#FDEAE9' },
                        },
                        accent: { 100: '#E650C5', 85: '#EC73CD', 50: '#F8CDEF', 20: '#FDEDFC' },
                        shadedOfGray: { 100: '#000000', 60: '#666666', 40: '#999999', 30: '#B2B2B2', 20: '#CCCCCC', 10: '#E6E6E6' },
                        white: '#FFFFFF',
                    },
                    fontFamily: {
                        'primary': ['Poppins', 'sans-serif'],
                        'secondaryAndButton': ['Urbanist', 'sans-serif'],
                    }
                }
            }
        }
    </script>

    <style>
        /* Global Styles Force Dark Theme */
        body { font-family: 'Urbanist', sans-serif; }
        h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif; }

        /* Custom Dark Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #020A36; }
        ::-webkit-scrollbar-thumb { background: #142C80; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #4F6CC3; }
    </style>
</head>

{{-- BODY: Force bg-primary-100 (Dark) --}}
<body class="bg-primary-100 text-white flex h-screen overflow-hidden antialiased selection:bg-secondary-happy-100 selection:text-white">

    {{-- Mobile Overlay --}}
    <div id="mobile-overlay" class="fixed inset-0 bg-[#020a36]/90 backdrop-blur-sm z-20 hidden lg:hidden transition-opacity opacity-0"></div>

    {{-- Include Sidebar --}}
    @include('components.superadmin.sidebar')

    <div class="flex-1 flex flex-col h-screen overflow-hidden relative">

        {{-- Include Navbar --}}
        @include('components.superadmin.navbar')

        {{-- CONTENT AREA --}}
        <main class="flex-1 overflow-y-auto p-4 lg:p-8 bg-primary-100 scroll-smooth relative">

            {{-- Dekorasi Gradient Background --}}
            <div class="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-primary-85/50 to-transparent pointer-events-none z-0"></div>

            <div class="relative z-10 min-h-[85vh]">
                @yield('content')
            </div>

            {{-- Footer Simple --}}
            <div class="mt-12 py-6 border-t border-primary-70 text-center md:text-right text-[#9CA3AF] text-xs relative z-10">
                <p>&copy; {{ date('Y') }} Roodio SuperAdmin.</p>
            </div>
        </main>
    </div>

    {{-- Script Toggle Sidebar --}}
    <script>
        const sidebar = document.getElementById('sidebar');
        const openSidebarBtn = document.getElementById('open-sidebar');
        const closeSidebarBtn = document.getElementById('close-sidebar');
        const overlay = document.getElementById('mobile-overlay');

        function toggleSidebar() {
            const isClosed = sidebar.classList.contains('-translate-x-full');
            if (isClosed) {
                sidebar.classList.remove('-translate-x-full');
                overlay.classList.remove('hidden');
                setTimeout(() => overlay.classList.remove('opacity-0'), 10);
            } else {
                sidebar.classList.add('-translate-x-full');
                overlay.classList.add('opacity-0');
                setTimeout(() => overlay.classList.add('hidden'), 300);
            }
        }

        if(openSidebarBtn) openSidebarBtn.addEventListener('click', toggleSidebar);
        if(closeSidebarBtn) closeSidebarBtn.addEventListener('click', toggleSidebar);
        if(overlay) overlay.addEventListener('click', toggleSidebar);
    </script>
</body>
</html>
