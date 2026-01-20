<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>@yield('title', 'Super Admin Panel')</title>

    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: { 100: '#020A36', 85: '#06134D', 70: '#0D1F67', 60: '#142C80', 50: '#1F3A98', 30: '#4F6CC3', 20: '#7591DB', 10: '#A4BEF2' },
                        secondary: {
                            happy: { 100: '#FF8E2B', 20: '#FFF2E5' },
                            sad: { 100: '#6A4FBF', 20: '#EEE8FB' },
                            relaxed: { 100: '#28C76F', 20: '#E0F7EB' },
                            angry: { 100: '#E63946', 20: '#FDEAE9' },
                        },
                        accent: { 100: '#E650C5', 85: '#EC73CD', 50: '#F8CDEF', 20: '#FDEDFC' },
                        shadedOfGray: { 100: '#000000', 60: '#666666', 40: '#999999', 20: '#CCCCCC', 10: '#E6E6E6' },
                    },
                    fontFamily: {
                        'primary': ['Poppins', 'sans-serif'],
                        'secondaryAndButton': ['Aeonik', 'sans-serif'],
                    }
                }
            }
        }
    </script>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">

    <style>
        body { font-family: 'Aeonik', sans-serif; }
        .font-primary { font-family: 'Poppins', sans-serif; }
    </style>
</head>
<body class="bg-primary-10/30 text-primary-100 flex h-screen overflow-hidden">

    <div id="mobile-overlay" class="fixed inset-0 bg-black/50 z-20 hidden lg:hidden transition-opacity opacity-0"></div>

    @include('components.superadmin.sidebar')

    <div class="flex-1 flex flex-col h-screen overflow-hidden relative">

        @include('components.superadmin.navbar')

        <main class="flex-1 overflow-y-auto p-4 lg:p-8 bg-primary-10/30">
            @yield('content')
        </main>
    </div>

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
