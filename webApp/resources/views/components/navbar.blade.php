@props([
    'mood'
])


@php
    $contrastStyle = [
        'happy' => 'bg-secondary-happy-85',
        'sad' => 'bg-secondary-sad-85',
        'relaxed' => 'bg-secondary-relaxed-85',
        'angry' => 'bg-secondary-angry-85'
    ];

    $softStyle = [
        'happy' => 'bg-secondary-happy-10',
        'sad' => 'bg-secondary-sad-10',
        'relaxed' => 'bg-secondary-relaxed-10',
        'angry' => 'bg-secondary-angry-10'
    ];

    $textStyle = [
        'happy' => 'text-secondary-happy-50',
        'sad' => 'text-secondary-sad-50',
        'relaxed' => 'text-secondary-relaxed-50',
        'angry' => 'text-secondary-angry-50'
    ];
@endphp


<div class='w-full h-16 bg-primary-85 flex justify-center items-center border-b-[0.5px] border-white'>
    <div class='w-full flex flex-row justify-between items-center px-4'>
        <div class='w-60 flex flex-row items-start gap-3'>
            <x-iconButton type='hamburger' :mood='$mood'></x-iconButton>
            <div class='w-40'>
                {{-- Route is not fixed --}}
                <a href="{{ route('user.index') }}" class='cursor-default'>
                    <img src="{{ asset('assets/logo/logo-horizontal.png') }}" alt="logo">
                </a>
            </div>
        </div>
        <div class='w-xl h-max invisible lg:visible relative transition-transform duration-500 ease-out group' id='searchbar'>
            <x-input type='search' :mood='$mood' id='search' placeholder='Search songs, artists, lyrics'></x-input>
            <div class='absolute top-1/2 right-4 flex flex-row -translate-y-1/2 opacity-50 group-focus-within:hidden' id='searchContent'>
                <div class='border-[0.5px] border-shadedOfGray-50 bg-shadedOfGray-20 font-secondaryAndButton px-1 py-[0.25px] rounded-md text-micro text-shadedOfGray-85'>CTRL</div>
                <p class='mx-1'>+</p>
                <div class='border-[0.5px] border-shadedOfGray-50 bg-shadedOfGray-20 font-secondaryAndButton px-1 py-[0.25px] rounded-md text-micro text-shadedOfGray-85'>K</div>
            </div>
        </div>
        <div class=' flex flex-row items-center justify-end gap-5'>
            <div class='w-8 h-8 lg:hidden cursor-pointer' id='searchIcon'>
                <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

                <!-- Uploaded to: SVG Repo, www.svgrepo.com, Transformed by: SVG Repo Mixer Tools -->
                <svg width="30px" height="30px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">

                <g id="SVGRepo_bgCarrier" stroke-width="0"/>

                <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"/>

                <g id="SVGRepo_iconCarrier"> <path d="M11 6C13.7614 6 16 8.23858 16 11M16.6588 16.6549L21 21M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/> </g>

                </svg>
            </div>
            <div class='flex flex-row items-center cursor-pointer' id='currentMood'>
                <p class='text-primary-70 font-secondaryAndButton text-small {{ $softStyle[$mood] }} pr-5 relative -right-3 pl-2 rounded-md py-0.5 hidden lg:block'>{{ Str::ucfirst($mood) }}</p>
                <div class='w-10 h-10 rounded-full p-0.5 relative z-5 {{ $contrastStyle[$mood] }}'>
                    <img src="{{ asset('assets/moods/'. $mood .'.png') }}" alt="mood" class='drop-shadow-lg drop-shadow-white'>
                </div>
            </div>
            <div class='flex flex-row gap-1.75 items-center justify-between w-max cursor-pointer' id='profileNavbar'>
                <div class='flex-col text-white hidden md:flex'>
                    <p class='text-small font-bold {{ $textStyle[$mood] }}'>{{ Str::limit($user->userDetail->fullname, 12) }}</p>
                    <p class='text-micro'>{{ '@' . Str::limit($user->username, 9) }}</p>
                </div>
                <div class='w-10 h-10 bg-primary-10 rounded-full flex items-center justify-center relative z-5 overflow-hidden'>
                    @if (isset($user->userDetail->profilePhoto))
                        <img src="{{ config('filesystems.disks.azure.url') . '/' . $user->userDetail->profilePhoto }}" alt="{{ $user->userDetail->fullname }}" class='w-full h-full object-cover'> 
                    @else
                        <p class='text-paragraph font-primary font-bold text-primary-70 h-fit'>{{ Str::charAt(Str::upper($user->userDetail->fullname), 0) }}</p>
                    @endif
                </div>
            </div>
        </div>
    </div>
</div>
