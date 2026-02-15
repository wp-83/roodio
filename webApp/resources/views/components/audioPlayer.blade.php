@props([
    'mood',
    'title' => 'Title',
    'artist' => 'Artist'
])


@php
    $mainBtnStyle = [
        'happy' => 'bg-secondary-happy-85',
        'sad' => 'bg-secondary-sad-85',
        'relaxed' => 'bg-secondary-relaxed-85',
        'angry' => 'bg-secondary-angry-85'
    ];

    $bgStyle = [
        'happy' => 'hover:bg-secondary-happy-10/30',
        'sad' => 'hover:bg-secondary-sad-10/30',
        'relaxed' => 'hover:bg-secondary-relaxed-10/30',
        'angry' => 'hover:bg-secondary-angry-10/30'
    ];

    $textStyle = [
        'happy' => 'text-secondary-happy-30',
        'sad' => 'text-secondary-sad-30',
        'relaxed' => 'text-secondary-relaxed-30',
        'angry' => 'text-secondary-angry-30'
    ];

    $elementStyle = [
        'happy' => '#FFE9D3',
        'sad' => '#E2D9F7',
        'relaxed' => '#CCF2DD',
        'angry' => '#FBDADC'
    ];

    $sliderStyle = [
        'happy' => 'accent-secondary-happy-70',
        'sad' => 'accent-secondary-sad-70',
        'relaxed' => 'accent-secondary-relaxed-70',
        'angry' => 'accent-secondary-angry-70',
    ];

    $audioControlOptions = ['loop', 'shuffle', 'speaker-muted'];
    $audioControlLabel = [
        'loop' => 'Loop Song',
        'shuffle' => 'Shuffle Play',
        'speaker-muted' => 'Muted Audio'
    ];

    $hoverBgMoodStyle = [
        'happy' => 'hover:bg-secondary-happy-20',
        'sad' => 'hover:bg-secondary-sad-20',
        'relaxed' => 'hover:bg-secondary-relaxed-20',
        'angry' => 'hover:bg-secondary-angry-20'
    ];

    // Soft background for Active State (Same as hover but persistent)
    $activeBgStyle = [
        'happy' => 'bg-secondary-happy-20',
        'sad' => 'bg-secondary-sad-20',
        'relaxed' => 'bg-secondary-relaxed-20',
        'angry' => 'bg-secondary-angry-20'
    ];
@endphp


<x-modal modalId='audioControlPopup' additionalStyle='absolute right-5 bottom-24' :isNeedBg='true'>
    <x-slot name='body'>
        <div class='w-full flex flex-col gap-2.5 font-secondaryAndButton text-small'>
            @foreach ($audioControlOptions as $audioCtrlOpt)
            @php
                $optId = $audioCtrlOpt === 'speaker-muted' ? 'speaker-mobile' : $audioCtrlOpt . '-mobile';
            @endphp
            <button id="{{ $optId }}" 
                class='w-full text-left cursor-pointer h-max rounded-md px-2 py-1.5 flex flex-row items-center gap-2.5 transition-all duration-200' 
                data-active-class="{{ $activeBgStyle[$mood] }}" 
                data-inactive-class="bg-transparent {{ $hoverBgMoodStyle[$mood] }}">
                <img src="{{ asset('assets/icons/' . $audioCtrlOpt . '.svg') }}" alt="{{ $audioCtrlOpt }}" class='w-7 h-7 pointer-events-none'>
                <p class='text-primary-60 pointer-events-none'>{{ Str::ucfirst($audioControlLabel[$audioCtrlOpt]) }}</p>
            </button>
            @endforeach
        </div>
    </x-slot>
</x-modal>

<div 
    id='audioPlayer' 
    class='w-full shrink-0' 
    x-data="{
        isExpanded: false,
        songTitle: '{{ $title }}',
        songArtist: '{{ $artist }}',
        songImage: '',
        init() {
            window.addEventListener('open-player-popup', () => { this.isExpanded = true; });
            window.addEventListener('close-player-popup', () => { this.isExpanded = false; });
            window.addEventListener('song-changed', (e) => {
                this.songTitle = e.detail.title;
                this.songArtist = e.detail.artist;
                this.songImage = e.detail.image;
            });
            
            // Initial sync from DOM if needed
            const img = document.getElementById('playerImage');
            if(img) this.songImage = img.src;
            const ttl = document.getElementById('playerTitle');
            if(ttl) this.songTitle = ttl.innerText;
            const art = document.getElementById('playerArtist');
            if(art) this.songArtist = art.innerText;
            
            // Close popup on navigation
            document.addEventListener('livewire:navigated', () => { this.isExpanded = false; });
        }
    }">

    <!-- Full Screen Popup -->
    <div
        class="fixed right-0 left-0 md:left-[5rem] bottom-[5.75rem] bg-primary-100 transition-all duration-500 ease-in-out flex flex-col overflow-hidden"
        :class="isExpanded ? 'top-16 opacity-100 visible' : 'top-full opacity-0 invisible'"
        style="z-index: 5;"
    >
        <div class="flex-1 flex flex-col md:flex-row min-h-0 overflow-y-auto md:overflow-hidden scrollbar-hide">
            <!-- Left Section (Song Info & Vinyl Disc) -->
            <div class="w-full md:w-3/5 relative aspect-square md:aspect-none max-h-[60vh] md:max-h-none md:min-h-0 shrink-0 overflow-hidden">
                <!-- Box 1/2 of section (Info Card) -->
                <div class="w-1/2 h-full py-2 md:py-4 md:px-0">
                    <div class="w-full h-full rounded-r-xl md:rounded-xl border-r border-primary-70 bg-black/30 flex flex-col overflow-hidden">
                        <!-- Title & Artist - top right -->
                        <div class="text-right p-4 shrink-0">
                            <h2 class="text-lg font-bold text-white truncate" x-text="songTitle"></h2>
                            <p class="text-sm text-primary-40 truncate" x-text="songArtist"></p>
                        </div>
                        <!-- Cover Image -->
                        <div class="flex-1 min-h-0 px-4 pb-4">
                            <img :src="songImage" alt="" class="w-full h-full object-cover rounded-lg">
                        </div>
                    </div>
                </div>
                <!-- Vinyl Circle - spinning with beat visualization -->
                <!-- Container enlarged to allow overflow visualization, clipped at 50% of THIS container -->
                <div class="absolute inset-0 w-full h-full flex items-center justify-center md:h-auto md:w-[85%] md:left-[50%] md:-translate-x-1/2 md:top-1/2 md:-translate-y-1/2 md:aspect-square" style="clip-path: inset(0 0 0 50%);">
                    <!-- Vinyl Disc (approx 56% original size -> 66% of 85%) -->
                    <div id="vinylDisc" class="h-[80%] aspect-square md:w-[66%] md:h-[66%] md:aspect-none rounded-full border-[6px] border-primary-40/30 relative vinyl-spin vinyl-paused">
                        <!-- Canvas for circular beat visualization - Enlarged to radiate OUTWARDS -->
                        <canvas id="audioVisualizer" class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[210%] h-[210%] rounded-full pointer-events-none"></canvas>
                        <!-- Album art center -->
                        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[40%] aspect-square rounded-full overflow-hidden border-2 border-primary-40/30 z-10">
                            <img :src="songImage" alt="" class="w-full h-full object-cover">
                        </div>
                        <!-- Center hole -->
                        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[8%] aspect-square rounded-full bg-primary-100 z-20"></div>
                    </div>
                </div>

                <style>
                    @keyframes vinyl-rotate {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                    .vinyl-spin {
                        animation: vinyl-rotate 4s linear infinite;
                    }
                    .vinyl-paused {
                        animation-play-state: paused;
                    }
                    .vinyl-playing {
                        animation-play-state: running;
                    }
                </style>
            </div>
            <!-- Right Section (Tracks & Lyrics) -->
            <div class="w-full md:w-2/5 border-t md:border-t-0 md:border-l border-primary-70 flex flex-col min-h-[40vh] md:min-h-0" x-data="{ activeTab: 'lyrics' }">
                <!-- Tab Headers -->
                <div class="flex flex-row shrink-0 border-b border-primary-70">
                    <button
                        @click="activeTab = 'tracks'"
                        :class="activeTab === 'tracks' ? 'border-b-2 border-white text-white' : 'text-primary-40 hover:text-white'"
                        class="flex-1 py-3 text-center font-secondaryAndButton text-body-size transition-colors duration-200"
                    >Tracks</button>
                    <button
                        @click="activeTab = 'lyrics'"
                        :class="activeTab === 'lyrics' ? 'border-b-2 border-white text-white' : 'text-primary-40 hover:text-white'"
                        class="flex-1 py-3 text-center font-secondaryAndButton text-body-size transition-colors duration-200"
                    >Lyrics</button>
                </div>
                <!-- Tab Content -->
                <div class="flex-1 md:overflow-y-auto p-4 scrollbar scrollbar-thumb-primary-10/75 scrollbar-track-transparent">
                    <!-- Lyrics Content -->
                    <div x-show="activeTab === 'lyrics'" class="w-full md:h-full">
                        <div id="popupLyricsContent" class="text-primary-20 text-small font-secondaryAndButton whitespace-pre-line leading-relaxed">
                            <!-- Lyrics will be rendered here by JS -->
                        </div>
                    </div>
                    <!-- Tracks Content -->
                    <div x-show="activeTab === 'tracks'" class="w-full md:h-full">
                        <div id="popupTracksList" class="flex flex-col gap-1">
                            <!-- Tracks will be rendered here by JS -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Overlay for no song -->
    <div id='overlayNan' class='hidden absolute bottom-0 z-10 h-23.25 w-full bg-shadedOfGray-85/50 cursor-not-allowed'></div>
    
    <!-- Audio Element -->
    <audio id='audio' preload="metadata" crossorigin="anonymous"></audio>
    
    <!-- Main Player Bar -->
    <div class='relative bg-primary-85 w-full'>
        <!-- Progress Bar Container -->
        <!-- Progress Bar Container -->
        <div 
            id="progressContainer" 
            class="absolute -top-2 left-0 w-full h-4 bg-transparent cursor-pointer flex items-center group z-20 outline-none" 
            tabindex="0"
            role="slider" 
            aria-label="Seek Slider" 
            aria-valuemin="0" 
            aria-valuemax="100" 
            aria-valuenow="0"
        >
            <div class="w-full h-1.25 bg-white group-hover:h-2 transition-all duration-200 pointer-events-none">
                <div id="progressBar" class="{{ 'h-full w-0 ' . $mainBtnStyle[$mood] . ' ' }}"></div>
            </div>
        </div>
        
        <!-- Player Controls -->
        <div class='w-full h-22 flex flex-row items-center justify-between px-5'>
            <div class='flex flex-row items-center gap-2'>
                <div class='h-14 w-14 bg-shadedOfGray-20 rounded-md overflow-hidden shrink-0'>
                    <img src="" alt="music" id="playerImage" class="w-full h-full object-cover">
                </div>
                <div class='hidden md:block text-white font-secondaryAndButton truncate max-w-[120px] xs:max-w-[150px] lg:max-w-[200px]'>
                    <p class='{{ 'text-body-size font-bold ' . $textStyle[$mood] . ' truncate' }}' id="playerTitle">{{ Str::limit($title, 35) }}</p>
                    <p class='text-micro truncate' id="playerArtist">{{ Str::limit($artist, 30) }}</p>
                </div>
            </div>
            
            <div class='flex flex-row items-center gap-4 absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2'>
                <button class='{{ 'w-9 h-9 p-1 rounded-full cursor-pointer ' . $bgStyle[$mood] . ' ' }}' id='prev'>
                    <svg fill="{{ $elementStyle[$mood] }}" width="100%" height="100%" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path fill-rule="evenodd" d="M18.4265377,4.18076808 L8.42653766,11.1807681 C7.85782078,11.5788699 7.85782078,12.4211301 8.42653766,12.8192319 L18.4265377,19.8192319 C19.0893151,20.2831762 20,19.809023 20,19 L20,5 C20,4.19097699 19.0893151,3.71682385 18.4265377,4.18076808 Z M5,4 C4.44771525,4 4,4.44771525 4,5 L4,19 C4,19.5522847 4.44771525,20 5,20 C5.55228475,20 6,19.5522847 6,19 L6,5 C6,4.44771525 5.55228475,4 5,4 Z M18,6.92065556 L18,17.0793444 L10.7437937,12 L18,6.92065556Z"/> 
                    </svg>
                </button>
                
                <div>
                    <button class='{{ 'w-15 h-15 p-2 rounded-full cursor-pointer hidden ' . $mainBtnStyle[$mood] . ' ' }}' id='pause'>
                        <svg width="100%" height="100%" viewBox="-0.5 0 25 25" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M10 6.42004C10 4.76319 8.65685 3.42004 7 3.42004C5.34315 3.42004 4 4.76319 4 6.42004V18.42C4 20.0769 5.34315 21.42 7 21.42C8.65685 21.42 10 20.0769 10 18.42V6.42004Z" stroke="{{ $elementStyle[$mood] }}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/> 
                            <path d="M20 6.42004C20 4.76319 18.6569 3.42004 17 3.42004C15.3431 3.42004 14 4.76319 14 6.42004V18.42C14 20.0769 15.3431 21.42 17 21.42C18.6569 21.42 20 20.0769 20 18.42V6.42004Z" stroke="{{ $elementStyle[$mood] }}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/> 
                        </svg>
                    </button>
                    <button class='{{ 'w-15 h-15 p-2 rounded-full cursor-pointer ' . $mainBtnStyle[$mood] . ' ' }}' id='play'>
                        <svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M5.46484 3.92349C4.79896 3.5739 4 4.05683 4 4.80888V19.1911C4 19.9432 4.79896 20.4261 5.46483 20.0765L19.1622 12.8854C19.8758 12.5108 19.8758 11.4892 19.1622 11.1146L5.46484 3.92349ZM2 4.80888C2 2.55271 4.3969 1.10395 6.39451 2.15269L20.0919 9.34382C22.2326 10.4677 22.2325 13.5324 20.0919 14.6562L6.3945 21.8473C4.39689 22.8961 2 21.4473 2 19.1911V4.80888Z" fill="{{ $elementStyle[$mood] }}"/> 
                        </svg>
                    </button>
                </div>
                
                <button class='{{ 'w-9 h-9 p-1 rounded-full cursor-pointer ' . $bgStyle[$mood] . ' ' }}' id='next'>
                    <svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M15.3371,12.4218 L5.76844,18.511 C5.43558,18.7228 5,18.4837 5,18.0892 L5,5.91084 C5,5.51629 5.43558,5.27718 5.76844,5.48901 L15.3371,11.5782 C15.6459,11.7746 15.6459,12.2254 15.3371,12.4218 Z" stroke="{{ $elementStyle[$mood] }}" stroke-width="2" stroke-linecap="round" fill="none"/>
                        <line x1="19" y1="5" x2="19" y2="19" stroke="{{ $elementStyle[$mood] }}" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            </div>
            
            <div class='flex flex-row gap-3 items-center'>
                <div class='hidden md:flex flex-row items-center text-white text-small'>
                    <span id='currentDuration'>--:--</span>
                    <span class="mx-0.5">/</span>
                    <span id='duration'>--:--</span>
                </div>
                <div class='flex-row hidden lg:flex gap-2 items-center'>
                    <button class='{{ 'w-9 h-9 p-1 rounded-full cursor-pointer ' . $bgStyle[$mood] . ' ' }}' id='loop' data-active-class="{{ $activeBgStyle[$mood] }}" data-inactive-class="{{ $bgStyle[$mood] }}">
                        <svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M18 4L21 7M21 7L18 10M21 7H7C4.79086 7 3 8.79086 3 11M6 20L3 17M3 17L6 14M3 17H17C19.2091 17 21 15.2091 21 13" stroke="{{ $elementStyle[$mood] }}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/> 
                        </svg>
                    </button>
                    <button class='{{ 'w-9 h-9 p-1 rounded-full cursor-pointer ' . $bgStyle[$mood] . ' ' }}' id='shuffle' data-active-class="{{ $activeBgStyle[$mood] }}" data-inactive-class="{{ $bgStyle[$mood] }}">
                        <svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M16.4697 9.46967C16.1768 9.76256 16.1768 10.2374 16.4697 10.5303C16.7626 10.8232 17.2374 10.8232 17.5303 10.5303L16.4697 9.46967ZM19.5303 8.53033C19.8232 8.23744 19.8232 7.76256 19.5303 7.46967C19.2374 7.17678 18.7626 7.17678 18.4697 7.46967L19.5303 8.53033ZM18.4697 8.53033C18.7626 8.82322 19.2374 8.82322 19.5303 8.53033C19.8232 8.23744 19.8232 7.76256 19.5303 7.46967L18.4697 8.53033ZM17.5303 5.46967C17.2374 5.17678 16.7626 5.17678 16.4697 5.46967C16.1768 5.76256 16.1768 6.23744 16.4697 6.53033L17.5303 5.46967ZM19 8.75C19.4142 8.75 19.75 8.41421 19.75 8C19.75 7.58579 19.4142 7.25 19 7.25V8.75ZM16.7 8L16.6993 8.75H16.7V8ZM12.518 10.252L13.1446 10.6642L13.1446 10.6642L12.518 10.252ZM10.7414 11.5878C10.5138 11.9338 10.6097 12.3989 10.9558 12.6266C11.3018 12.8542 11.7669 12.7583 11.9946 12.4122L10.7414 11.5878ZM11.9946 12.4122C12.2222 12.0662 12.1263 11.6011 11.7802 11.3734C11.4342 11.1458 10.9691 11.2417 10.7414 11.5878L11.9946 12.4122ZM10.218 13.748L9.59144 13.3358L9.59143 13.3358L10.218 13.748ZM6.041 16V16.75H6.04102L6.041 16ZM5 15.25C4.58579 15.25 4.25 15.5858 4.25 16C4.25 16.4142 4.58579 16.75 5 16.75V15.25ZM11.9946 11.5878C11.7669 11.2417 11.3018 11.1458 10.9558 11.3734C10.6097 11.6011 10.5138 12.0662 10.7414 12.4122L11.9946 11.5878ZM12.518 13.748L13.1446 13.3358L13.1446 13.3358L12.518 13.748ZM16.7 16V15.25H16.6993L16.7 16ZM19 16.75C19.4142 16.75 19.75 16.4142 19.75 16C19.75 15.5858 19.4142 15.25 19 15.25V16.75ZM10.7414 12.4122C10.9691 12.7583 11.4342 12.8542 11.7802 12.6266C12.1263 12.3989 12.2222 11.9338 11.9946 11.5878L10.7414 12.4122ZM10.218 10.252L9.59143 10.6642L9.59144 10.6642L10.218 10.252ZM6.041 8L6.04102 7.25H6.041V8ZM5 7.25C4.58579 7.25 4.25 7.58579 4.25 8C4.25 8.41421 4.58579 8.75 5 8.75V7.25ZM17.5303 13.4697C17.2374 13.1768 16.7626 13.1768 16.4697 13.4697C16.1768 13.7626 16.1768 14.2374 16.4697 14.5303L17.5303 13.4697ZM18.4697 16.5303C18.7626 16.8232 19.2374 16.8232 19.5303 16.5303C19.8232 16.2374 19.8232 15.7626 19.5303 15.4697L18.4697 16.5303ZM19.5303 16.5303C19.8232 16.2374 19.8232 15.7626 19.5303 15.4697C19.2374 15.1768 18.7626 15.1768 18.4697 15.4697L19.5303 16.5303ZM16.4697 17.4697C16.1768 17.7626 16.1768 18.2374 16.4697 18.5303C16.7626 18.8232 17.2374 18.8232 17.5303 18.5303L16.4697 17.4697ZM17.5303 10.5303L19.5303 8.53033L18.4697 7.46967L16.4697 9.46967L17.5303 10.5303ZM19.5303 7.46967L17.5303 5.46967L16.4697 6.53033L18.4697 8.53033L19.5303 7.46967ZM19 7.25H16.7V8.75H19V7.25ZM16.7007 7.25C14.7638 7.24812 12.956 8.22159 11.8914 9.8398L13.1446 10.6642C13.9314 9.46813 15.2676 8.74861 16.6993 8.75L16.7007 7.25ZM11.8914 9.83979L10.7414 11.5878L11.9946 12.4122L13.1446 10.6642L11.8914 9.83979ZM10.7414 11.5878L9.59144 13.3358L10.8446 14.1602L11.9946 12.4122L10.7414 11.5878ZM9.59143 13.3358C8.80541 14.5306 7.47115 15.25 6.04098 15.25L6.04102 16.75C7.97596 16.7499 9.78113 15.7767 10.8446 14.1602L9.59143 13.3358ZM6.041 15.25H5V16.75H6.041V15.25ZM10.7414 12.4122L11.8914 14.1602L13.1446 13.3358L11.9946 11.5878L10.7414 12.4122ZM11.8914 14.1602C12.956 15.7784 14.7638 16.7519 16.7007 16.75L16.6993 15.25C15.2676 15.2514 13.9314 14.5319 13.1446 13.3358L11.8914 14.1602ZM16.7 16.75H19V15.25H16.7V16.75ZM11.9946 11.5878L10.8446 9.83979L9.59144 10.6642L10.7414 12.4122L11.9946 11.5878ZM10.8446 9.8398C9.78113 8.2233 7.97596 7.25005 6.04102 7.25L6.04098 8.75C7.47115 8.75004 8.80541 9.46939 9.59143 10.6642L10.8446 9.8398ZM6.041 7.25H5V8.75H6.041V7.25ZM16.4697 14.5303L18.4697 16.5303L19.5303 15.4697L17.5303 13.4697L16.4697 14.5303ZM18.4697 15.4697L16.4697 17.4697L17.5303 18.5303L19.5303 16.5303L18.4697 15.4697Z" fill="{{ $elementStyle[$mood] }}"/> 
                        </svg>
                    </button>
                    <div class='flex gap-1 items-center'>
                        <button class='{{ 'w-9 h-9 p-1 rounded-full cursor-pointer ' . $bgStyle[$mood] . ' ' }}' id='speaker'>
                            <svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M19 6C20.5 7.5 21 10 21 12C21 14 20.5 16.5 19 18M16 8.99998C16.5 9.49998 17 10.5 17 12C17 13.5 16.5 14.5 16 15M3 10.5V13.5C3 14.6046 3.5 15.5 5.5 16C7.5 16.5 9 21 12 21C14 21 14 3 12 3C9 3 7.5 7.5 5.5 8C3.5 8.5 3 9.39543 3 10.5Z" stroke="{{ $elementStyle[$mood] }}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> 
                            </svg>
                        </button>
                        <button class='{{ 'w-9 h-9 p-1 rounded-full cursor-pointer hidden ' . $bgStyle[$mood] . ' ' }}' id='muted'>
                            <svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M22 9L16 15M16 9L22 15M3 10.5V13.5C3 14.6046 3.5 15.5 5.5 16C7.5 16.5 9 21 12 21C14 21 14 3 12 3C9 3 7.5 7.5 5.5 8C3.5 8.5 3 9.39543 3 10.5Z" stroke="{{ $elementStyle[$mood] }}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/> 
                            </svg>
                        </button>
                        <input type="range" name="volume" id="volumeSlider" min='0' max='1' step='0.01' class='{{ 'w-20 lg:w-28 ' . $sliderStyle[$mood] . ' ' }}'>
                    </div>
                    <!-- Expand Button -->
                    <button class="flex items-center justify-center cursor-pointer p-0.5 rounded-full hover:bg-white/10 transition-all h-9 w-9" @click="isExpanded = !isExpanded" title="Expand Player">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" :class="isExpanded ? 'rotate-180' : ''" class="transition-transform duration-300">
                            <path d="M6 9L12 15L18 9" stroke="{{ $elementStyle[$mood] }}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                </div>
                    <!-- Mobile Expand/Collapse Arrow -->
                    <button class="flex lg:hidden items-center justify-center cursor-pointer p-0.5 rounded-full hover:bg-white/10 transition-all h-9 w-9" @click="isExpanded = !isExpanded" title="Toggle Player">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" :class="isExpanded ? 'rotate-180' : ''" class="transition-transform duration-300">
                            <path d="M6 9L12 15L18 9" stroke="{{ $elementStyle[$mood] }}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                    <x-iconButton type='kebab' :mood='$mood' class='lg:hidden' id='audioControlResponsive'></x-iconButton>
            </div>
        </div>
    </div>
</div>