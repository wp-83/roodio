@vite(['resources/css/app.css', 'resources/js/app.js'])

@props([

])

<div class='relative group'>
    <div class='relative z-10 bg-white w-max hover:bg-primary-10 px-3 py-2 rounded-2xl flex flex-col items-center duration-100'>
        <img src="{{ asset('assets/icons/home.svg') }}" alt="home" class='w-10 h-10'>
        <p>Home</p>
    </div>
    <div class='bg-secondary-happy-85 w-max h-max pl-24 px-2 py-2 absolute inset-0 top-1/2 -translate-y-1/2 -translate-x-full group-hover:translate-x-0 transition-transform duration-500'>
        <p>Beranda</p>
    </div>
</div>