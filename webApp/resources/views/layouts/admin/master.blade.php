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

    {{-- Tailwind CSS & Config --}}
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: {
                            100: '#020A36', // Background Utama (Deep Blue)
                            85: '#06134D',  // Sidebar / Card Surface
                            70: '#0D1F67',  // Border / Hover
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
                        accent: { 100: '#E650C5', 20: '#FDEDFC' },
                        shadedOfGray: {
                            100: '#000000', 60: '#666666', 40: '#9CA3AF', 30: '#B2B2B2', 20: '#CCCCCC', 10: '#E6E6E6',
                        },
                        white: '#FFFFFF',
                    },
                    fontFamily: {
                        'primary': ['Poppins', 'sans-serif'],
                        'secondaryAndButton': ['Urbanist', 'sans-serif'], // Fallback ke Urbanist jika Aeonik tidak ada
                    },
                    fontSize: {
                        'title': ['2.667rem', { lineHeight: '4rem' }],
                        'subtitle': ['2rem', { lineHeight: '3rem' }],
                        'body-size': ['1.167rem', { lineHeight: '1.75rem' }],
                        'small': ['1rem', { lineHeight: '1.5rem' }],
                        'micro': ['0.833rem', { lineHeight: '1.25rem' }],
                    },
                }
            }
        }
    </script>

    <style>
        /* Custom Scrollbar for Dark Theme */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #020A36; }
        ::-webkit-scrollbar-thumb { background: #142C80; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #4F6CC3; }

        body { font-family: 'Urbanist', sans-serif; }
        h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif; }
    </style>
</head>
<body class="bg-primary-100 text-white antialiased selection:bg-secondary-happy-100 selection:text-white overflow-hidden">

    <div class="flex h-screen w-full">

        {{-- SIDEBAR COMPONENT --}}
        @include('components.admin.sidebar')

        {{-- MAIN WRAPPER --}}
        <div class="flex flex-col flex-1 h-full min-w-0 overflow-hidden">

            {{-- NAVBAR COMPONENT --}}
            @include('components.admin.navbar')

            {{-- CONTENT SCROLL AREA --}}
            <main class="flex-1 overflow-y-auto overflow-x-hidden p-6 md:p-8 scroll-smooth relative">
                {{-- Background Gradient Decoration (Optional, like Roodio) --}}
                <div class="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-primary-85/50 to-transparent pointer-events-none z-0"></div>

                <div class="relative z-10">
                    @yield('content')
                </div>

                {{-- Footer --}}
                <div class="mt-12 py-6 border-t border-primary-70 text-center md:text-right text-shadedOfGray-40 text-micro relative z-10">
                    <p>&copy; {{ date('Y') }} Roodio Music. Admin Panel.</p>
                </div>
            </main>
        </div>
    </div>

</body>
</html>
