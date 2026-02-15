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

    $bgMoodStyle = [
        'happy' => 'bg-secondary-happy-20',
        'sad' => 'bg-secondary-sad-20',
        'relaxed' => 'bg-secondary-relaxed-20',
        'angry' => 'bg-secondary-angry-20'
    ];

    $textMoodStyle = [
        'happy' => 'text-secondary-happy-100',
        'sad' => 'text-secondary-sad-100',
        'relaxed' => 'text-secondary-relaxed-100',
        'angry' => 'text-secondary-angry-100'
    ];

    $hoverBgMoodStyle = [
        'happy' => 'hover:bg-secondary-happy-20',
        'sad' => 'hover:bg-secondary-sad-20',
        'relaxed' => 'hover:bg-secondary-relaxed-20',
        'angry' => 'hover:bg-secondary-angry-20'
    ];

    $checkboxStyle = [
        'happy' => 'accent-secondary-happy-50',
        'sad' => 'accent-secondary-sad-50',
        'relaxed' => 'accent-secondary-relaxed-50',
        'angry' => 'accent-secondary-angry-50'
    ];

    $moodOptions = ['happy', 'sad', 'relaxed', 'angry'];
@endphp


<x-modal modalId='profilePopup' additionalStyle='right-3 top-14 w-60 h-max '>
    <x-slot name='body'>
        <div class='absolute right-6 top-5' style='zoom: 0.75;' id='closeProfilePopup'>
            <x-iconButton :mood='$mood' type='cross'></x-iconButton>
        </div>
        <div class='flex flex-col items-center gap-2'>
            <div class='w-20 h-20 rounded-full flex items-center justify-center overflow-hidden {{ $bgMoodStyle[$mood] }}'>
                @if (!empty($profilePhoto))
                    <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" alt="{{ $fullname }}" class='w-full h-full object-cover'>
                @else
                <p class='text-title font-primary font-bold h-fit {{ $textMoodStyle[$mood] }}'>{{ Str::charAt(Str::upper($fullname), 0) }}</p>
                @endif
            </div>
            <div class='flex flex-col items-center'>
                <p class='text-small font-bold {{ $textMoodStyle[$mood] }}'>{{ Str::limit($fullname, 24) }}</p>
                <p class='text-micro text-primary-60'>{{ '@' . Str::limit($username, 18) }}</p>
            </div>
        </div>
        <hr class='my-2 border-primary-50'>
        <div class='w-full flex flex-col gap-2.5 font-secondaryAndButton text-small'>
            <a href="{{ route('user.profile') }}" wire:navigate>
                <div class='h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 {{ $hoverBgMoodStyle[$mood] }}'>
                    <img src="{{ asset('assets/icons/user.svg') }}" alt="user" class='w-7 h-7'>
                    <p class='text-primary-60'>Edit Your Profile</p>
                </div>
            </a>
            <form action="{{ route('auth.logout') }}" method="POST">
                @csrf
                <div onclick="this.closest('form').submit()"
                    class='cursor-pointer h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 {{ $hoverBgMoodStyle[$mood] }}'>

                    <img src="{{ asset('assets/icons/logout.svg') }}" alt="logout" class='w-7 h-7'>
                    <p class='text-primary-60'>Logout</p>
                </div>
            </form>
        </div>
    </x-slot>
</x-modal>

<x-modal modalId='changeMood' additionalStyle='right-20 top-14 md:right-48'>
    <x-slot name='body'>
        <form
            action="{{ route('mood.update') }}"
            method="POST"
        >
            @csrf

            <p class='mb-3 font-bold text-primary-60'>Change Your Mood</p>

            <div class='w-full flex flex-col gap-2.5 font-secondaryAndButton text-small'>
                @foreach ($moodOptions as $moodOpt)
                <button
                    type="submit"
                    name="mood"
                    value="{{ $moodOpt }}"
                    {{ ($moodOpt == $mood) ? 'disabled' : '' }}
                    class='w-full h-max rounded-sm px-2 py-1 flex flex-row items-center gap-2.5 text-left transition-colors duration-200
                    {{ (($moodOpt == $mood) ? $bgMoodStyle[$mood] . ' cursor-default opacity-80' : $hoverBgMoodStyle[$mood] . ' cursor-pointer') }}'
                >
                    <img src="{{ asset('assets/moods/' . $moodOpt . '.png') }}" alt="{{ $moodOpt }}" class='w-7 h-7'>
                    <p class='text-primary-60'>{{ Str::ucfirst($moodOpt) }}</p>
                </button>
                @endforeach
            </div>
        </form>

        <hr class='my-4 border-gray-200'>

        <form
            action="{{ route('preference.update') }}"
            method="POST"
            x-data="{
                isMatch: {{ (session('preferenceMood', 'match') == 'match') ? 'true' : 'false' }}
            }"
        >
            @csrf

            <p class='mb-3 font-bold text-primary-60'>Playlist Behaviour</p>

            <div class='flex flex-row items-center gap-1.25 w-max h-max'>

                <input type="hidden" name="preferenceMood" :value="isMatch ? 'match' : 'mismatch'">

                <input
                    type="checkbox"
                    id="playlistMood"
                    x-model="isMatch"
                    @change="$nextTick(() => $el.form.submit())"
                    class='w-6 h-6 rounded-lg border-2 border-gray-300 focus:ring-0 cursor-pointer {{ $checkboxStyle[$mood] }}'
                >

                <label for="playlistMood" class='text-micro md:text-small cursor-pointer text-primary-60'>
                    Based on mood
                </label>
            </div>
        </form>
    </x-slot>
</x-modal>

<div class='w-full h-16 bg-primary-85 flex justify-center items-center border-b-[0.5px] border-white'>
    <div class='w-full flex flex-row justify-between items-center px-4'>
        <div class='w-60 flex flex-row items-start gap-3'>
            <x-iconButton type='hamburger' :mood='$mood'></x-iconButton>
            <div class='w-40'>
                {{-- Route is not fixed --}}
                <a href="{{ route('user.index') }}" wire:navigate class='cursor-default'>
                    <img src="{{ asset('assets/logo/logo-horizontal.png') }}" alt="logo">
                </a>
            </div>
        </div>
        @if($showSearch)
        <form action="{{ url()->current() }}" method="GET" id="searchForm" class='w-xl h-max relative transition-transform duration-500 ease-out group'>
            <div class='hidden lg:flex flex-row items-center px-2 py-0.5 h-10 text-small rounded-full placeholder:text-micro placeholder:italic bg-shadedOfGray-10 hover:bg-white focus-within:bg-white ease-in-out duration-125 md:text-body-size md:placeholder:text-small md:h-9 group' id='searchbar'>
                <img src="{{ asset('assets/icons/search.svg') }}" alt="search" class='w-8 h-8 pr-1 mr-2 border-r-2 border-primary-70'>
                <input type="text" name="search" autocomplete="off" id='search' placeholder='Search...' class='w-full pr-2' value="{{ request('search') }}">
                <div onclick="document.getElementById('search').value=''; document.getElementById('searchForm').submit();" class='cursor-pointer {{ request('search') ? '' : 'invisible' }}' style='zoom:0.8;' id='searchClose'>
                    <x-iconButton type='cross' :mood='$mood'></x-iconButton>
                </div>
            </div>
            <div class='absolute top-1/2 right-4 hidden lg:flex flex-row -translate-y-1/2 opacity-50 group-focus-within:hidden {{ request('search') ? 'hidden' : '' }}' id='searchContent'>
                <div class='border-[0.5px] border-shadedOfGray-50 bg-shadedOfGray-20 font-secondaryAndButton px-1 py-[0.25px] rounded-md text-micro text-shadedOfGray-85'>CTRL</div>
                <p class='mx-1'>+</p>
                <div class='border-[0.5px] border-shadedOfGray-50 bg-shadedOfGray-20 font-secondaryAndButton px-1 py-[0.25px] rounded-md text-micro text-shadedOfGray-85'>K</div>
            </div>
        </form>
        <script>
            // Auto-submit search form on input change with debounce
            (function() {
                const searchInput = document.getElementById('search');
                const searchForm = document.getElementById('searchForm');
                let debounceTimer;
                
                if (searchInput && searchForm) {
                    searchInput.addEventListener('input', function() {
                        clearTimeout(debounceTimer);
                        debounceTimer = setTimeout(() => {
                            searchForm.submit();
                        }, 500);
                    });
                    
                    // Submit on Enter key
                    searchInput.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            e.preventDefault();
                            clearTimeout(debounceTimer);
                            searchForm.submit();
                        }
                    });
                }
            })();
        </script>
        @endif
        <div class=' flex flex-row items-center justify-end gap-5'>
            @if($showSearch)
            <div class='w-8 h-8 lg:hidden cursor-pointer' id='searchIcon'>
                <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

                    <!-- Uploaded to: SVG Repo, www.svgrepo.com, Transformed by: SVG Repo Mixer Tools -->
                    <svg width="30px" height="30px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">

                    <g id="SVGRepo_bgCarrier" stroke-width="0"/>

                    <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"/>

                    <g id="SVGRepo_iconCarrier"> <path d="M11 6C13.7614 6 16 8.23858 16 11M16.6588 16.6549L21 21M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/> </g>

                    </svg>
            </div>
            @endif
            <div class='flex flex-row items-center cursor-pointer' id='currentMood'>
                <p class='text-primary-70 font-secondaryAndButton text-small {{ $softStyle[$mood] }} pr-5 relative -right-3 pl-2 rounded-md py-0.5 hidden lg:block'>{{ Str::ucfirst($mood) }}</p>
                <div class='w-10 h-10 rounded-full p-0.5 relative z-5 {{ $contrastStyle[$mood] }}'>
                    <img src="{{ asset('assets/moods/'. $mood .'.png') }}" alt="mood" class='drop-shadow-lg drop-shadow-white'>
                </div>
            </div>
            <div class='flex flex-row gap-1.75 items-center justify-between w-max cursor-pointer' id='profileNavbar'>
                <div class='flex-col text-white hidden md:flex'>
                    <p class='text-small font-bold {{ $textStyle[$mood] }}'>{{ Str::limit($fullname, 12) }}</p>
                    <p class='text-micro'>{{ '@' . Str::limit($username, 9) }}</p>
                </div>
                <div class='w-10 h-10 bg-primary-10 rounded-full flex items-center justify-center relative z-5 overflow-hidden'>
                    @if (!empty($profilePhoto))
                        <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" alt="{{ $fullname }}" class='w-full h-full object-cover'>
                    @else
                        <p class='text-paragraph font-primary font-bold text-primary-70 h-fit'>{{ Str::charAt(Str::upper($fullname), 0) }}</p>
                    @endif
                </div>
            </div>
        </div>
    </div>
</div>
