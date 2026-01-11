@vite(['resources/css/app.css', 'resources/js/app.js'])

{{-- <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script> --}}

{{-- @props([

]) --}}

<html>
<head>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
</head>
<body class='overflow-hidden bg-primary-100 container-body'>
    <div class='w-full h-16 bg-primary-85 flex justify-center items-center border-b-[0.5px] border-white'>
        <div class='w-full flex flex-row justify-between items-center px-4'>
            <div class='w-60 flex flex-row items-start gap-3'>
                <x-iconButton type='hamburger' mood='happy'></x-iconButton>
                <div class='w-40'>
                    <a href="{{ route('awikwok') }}" class='cursor-default'>
                        <img src="{{ asset('assets/logo/logo-horizontal.png') }}" alt="logo">
                    </a>
                </div>
            </div>
            <div class='w-xl h-max invisible lg:visible relative'>
                <x-input type='search' id='search' placeholder='Search songs, artists, lyrics'></x-input>

            </div>
            <div class=' flex flex-row items-center justify-end gap-5'>
                <div class='w-8 h-8 lg:hidden'>
                    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

                    <!-- Uploaded to: SVG Repo, www.svgrepo.com, Transformed by: SVG Repo Mixer Tools -->
                    <svg width="30px" height="30px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">

                    <g id="SVGRepo_bgCarrier" stroke-width="0"/>

                    <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"/>

                    <g id="SVGRepo_iconCarrier"> <path d="M11 6C13.7614 6 16 8.23858 16 11M16.6588 16.6549L21 21M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/> </g>

                    </svg>
                </div>
                <div class='flex flex-row items-center'>
                    <p class='text-primary-70 font-secondaryAndButton text-small bg-secondary-happy-10 pr-5 relative -right-3 pl-2 rounded-md hidden lg:block'>Happy</p>
                    <div class='w-10 h-10 rounded-full bg-secondary-happy-100 p-0.5 relative z-5'>
                        <img src="{{ asset('assets/moods/happy.png') }}" alt="mood">
                    </div>
                </div>
                <div class='flex flex-row gap-1.75 items-center justify-between w-max'>
                    <div class='flex-col text-white hidden lg:flex'>
                        <p class='text-small'>{{ Str::limit('Andi Zulfikar', 12) }}</p>
                        <p class='text-micro'>{{ '@' . Str::limit('andikecebadai999', 9) }}</p>
                    </div>
                    <div class='w-10 h-10 bg-primary-10 rounded-full flex items-center justify-center relative z-5'>
                        <p class='text-paragraph font-primary font-bold text-primary-70 h-fit'>{{ Str::charAt(Str::upper('Andi'), 0) }}</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <x-sidebar mood='happy'></x-sidebar>
</body>
</html>

