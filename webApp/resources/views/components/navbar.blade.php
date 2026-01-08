@vite(['resources/css/app.css', 'resources/js/app.js'])

{{-- <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script> --}}

{{-- @props([

]) --}}

<html>
<head>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
</head>
<body class='overflow-hidden'>
    <div class='w-full h-14 bg-primary-70 flex justify-center items-center'>
        <div class='w-full flex flex-row justify-between items-center px-4'>
            <div class='w-60 flex flex-row items-start gap-3'>
                <x-iconButton type='hamburger' mood='angry'></x-iconButton>
                <div class='w-40'>
                    <a href="{{ route('awikwok') }}" class='cursor-default'>
                        <img src="{{ asset('assets/logo/logo-horizontal.png') }}" alt="logo">
                    </a>
                </div>
            </div>
            <div class='hidden'>
                <input type="search" name="" id="" class='w-xl h-8 outline-2 bg-white'>
            </div>
            <div class=' flex flex-row items-center justify-end gap-5'>
                <div class='w-8 h-8 bg-amber-900'>

                </div>
                <div class='flex flex-col hidden'>
                    <p>William Pratama</p>
                    <p>Online</p>
                </div>
                <div class='w-8 h-8'>
                    <img src="{{ asset('assets/defaults/user.png') }}" alt="">
                </div>
            </div>
        </div>
    </div>
    <x-sidebar mood='happy'></x-sidebar>
</body>
</html>

