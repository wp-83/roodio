@extends('layouts.master')


@section('title', 'ROODIO - Music Player Based on Your Mood')


@push('style')
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css" />
    <style>
        /* Custom Wave Separator */
        .wave-bottom {
            position: absolute;
            bottom: -5px; /* Aggressive overlap to hide gap */
            left: 0;
            width: 100%;
            overflow: hidden;
            line-height: 0;
            transform: none;
            z-index: 5;
        }
        .wave-bottom svg {
            position: relative;
            display: block;
            width: calc(100% + 1.3px);
            height: 60px;
        }
        @media (min-width: 768px) {
            .wave-bottom svg { height: 120px; }
        }
        .wave-bottom .shape-fill {
            fill: #F3F4F6; /* Matches bg-gray-100 */
        }

        .wave-top {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            overflow: hidden;
            line-height: 0;
            z-index: 2;
        }
        .wave-top svg {
            position: relative;
            display: block;
            width: calc(100% + 1.3px);
            height: 60px; /* Smaller wave on mobile */
        }
        @media (min-width: 768px) {
            .wave-top svg { height: 120px; }
        }
        .wave-top .shape-fill {
            fill: #F3F4F6; /* Matches bg-gray-100 */
        }

        /* Floating Animation */
        @keyframes float-slow {
            0% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(5deg); }
            100% { transform: translateY(0) rotate(0deg); }
        }
        @keyframes float-medium {
            0% { transform: translateY(0) translateX(0); }
            33% { transform: translateY(-30px) translateX(10px); }
            66% { transform: translateY(-10px) translateX(-10px); }
            100% { transform: translateY(0) translateX(0); }
        }
        @keyframes rotate-scale {
            0% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.1) rotate(180deg); }
            100% { transform: scale(1) rotate(360deg); }
        }
        @keyframes pulse-glow {
            0%, 100% { opacity: 0.1; transform: scale(1); }
            50% { opacity: 0.3; transform: scale(1.1); }
        }

        .animate-float-slow { animation: float-slow 8s ease-in-out infinite; }
        .animate-float-medium { animation: float-medium 6s ease-in-out infinite; }
        .animate-float-fast { animation: float-medium 4s ease-in-out infinite; }
        .animate-rotate-scale { animation: rotate-scale 20s linear infinite; }
        .animate-pulse-glow { animation: pulse-glow 4s ease-in-out infinite; }

        .shape-circle { border-radius: 50%; }
        .shape-triangle {
            width: 0; 
            height: 0; 
            border-left: 15px solid transparent;
            border-right: 15px solid transparent;
            border-bottom: 25px solid currentColor;
        }

        /* Specific Graphic Elements (Donuts & Dot Grids) */
        .shape-donut {
            border-radius: 50%;
            background: transparent;
            border-style: solid;
        }
        .bg-pattern-dots {
            background-image: radial-gradient(#ffffff 1px, transparent 1px);
            background-size: 60px 60px;
            opacity: 0.03;
        }
        .shape-dot-grid {
            background-image: radial-gradient(currentColor 2px, transparent 2px);
            background-size: 15px 15px;
        }

        .swiper-pagination-bullet-active {
            background-color: #06134D !important;
        }

        /* Carousel Scaling Effect */
        .featuresSwiper {
            padding-top: 50px !important;
            padding-bottom: 50px !important;
        }
        .swiper-slide {
            transition: all 0.4s ease;
            transform: scale(0.9); /* Slightly smaller inactive */
            opacity: 0.6; /* Reduced opacity */
            /* Removed blur */
        }
        .swiper-slide-active {
            transform: scale(1.05); /* Slightly larger active */
            opacity: 1;
            z-index: 10;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
    </style>
@endpush


@push('script')
    <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <script defer>
        document.addEventListener('DOMContentLoaded', function () {
            AOS.init({
                duration: 800,
                once: true,
                offset: 50,
                easing: 'ease-out-cubic'
            });

            var swiper = new Swiper(".featuresSwiper", {
                centeredSlides: true,
                loop: true,
                loopedSlides: 4,
                speed: 400,
                autoplay: {
                    delay: 5000,
                    disableOnInteraction: false,
                    pauseOnMouseEnter: true,
                },
                pagination: {
                    el: ".swiper-pagination",
                    clickable: true,
                },
                navigation: {
                    nextEl: ".swiper-button-next",
                    prevEl: ".swiper-button-prev",
                },
                observer: true,
                observeParents: true,
                observeSlideChildren: true,
                breakpoints: {
                    320: {
                        slidesPerView: 1.2,
                        slidesPerGroup: 1,
                        spaceBetween: 20,
                    },
                    640: {
                        slidesPerView: 1.8,
                        slidesPerGroup: 1,
                        spaceBetween: 30,
                    },
                    1024: {
                        slidesPerView: 2.5,
                        slidesPerGroup: 1,
                        spaceBetween: 40,
                    },
                },
            });
        });
    </script>
@endpush


@php
    $features = [
        [
            'icon' => '🎭',
            'title' => 'Smart Mood Recognition',
            'desc' => 'Roodio understands your mood and recommends music that truly matches how you feel.',
            'color' => 'angry'
        ],
        [
            'icon' => '📝',
            'title' => 'Mood Threads',
            'desc' => 'Share what you\'re feeling and express your emotions through interactive threads.',
            'color' => 'sad'
        ],
        [
            'icon' => '👥',
            'title' => 'Follow & Connect',
            'desc' => 'Follow others, gain followers, and discover music from people with similar vibes.',
            'color' => 'relaxed'
        ],
        [
            'icon' => '📊',
            'title' => 'Mood Insights',
            'desc' => 'View your mood recap weekly, monthly, or even yearly with clear summaries.',
            'color' => 'happy'
        ],
        [
            'icon' => '⏳',
            'title' => 'Sleep Timer',
            'desc' => 'Set a timer to automatically stop your music whenever you want.',
            'color' => 'sad'
        ],
        [
            'icon' => '⚡',
            'title' => 'Fast & Lightweight',
            'desc' => 'Smooth performance, fast loading, and optimized to stay light on your device.',
            'color' => 'relaxed'
        ],
    ];

    $advisors = [
        [
            'name' => 'Dr. Zulfany Erlisa Rasjid, B.Sc., MMSI.',
            'role' => 'Software Engineering Specialist',
            'photo' => 'zulfany.jpg'
        ],
        [
            'name' => 'Dr. Hidayaturrahman, S.Kom., M.T.',
            'role' => 'Machine Learning Specialist',
            'photo' => 'hidayaturrahman.jpg'
        ],
        [
            'name' => 'Francisco Maruli Panggabean, S.Kom., M.T.I.',
            'role' => 'Multimedia Systems Specialist',
            'photo' => 'francisco.jpg'
        ],
    ];

    $developers = [
        [
            'name' => 'Andi Zulfikar',
            'role' => 'Backend Developer',
            'photo' => 'andi.jpg'
        ],
        [
            'name' => 'William Pratama',
            'role' => 'Frontend Developer',
            'photo' => 'william.jpg'
        ],
        [
            'name' => 'Agnes Gonxha F. Sukma',
            'role' => 'UI/UX Designer',
            'photo' => 'agnes.jpg'
        ],
        [
            'name' => 'Felicia Wijaya',
            'role' => 'UI/UX Designer',
            'photo' => 'felicia.jpg'
        ],
        [
            'name' => 'Yoyada Indrayudha',
            'role' => 'Quality Assurance',
            'photo' => 'yoyada.jpg'
        ],
    ];
@endphp


@section('bodyContent')
    <div class="font-secondaryAndButton text-primary-85 bg-white overflow-x-hidden">
        
        <!-- navbar -->
        <header 
            x-data="{ scrolled: false }"
            @scroll.window="scrolled = window.scrollY > 0"
            :class="scrolled ? 'bg-primary-70/35' : 'bg-primary-70 shadow-xs shadow-white'"
            class="fixed w-full z-50 py-1.5 text-white transition-all duration-300">
            <div class="container mx-auto px-4 md:px-6 h-max flex items-center justify-between">
                <!-- Logo -->
                <a href="/" class="block w-24 md:w-36 lg:w-48 hover:opacity-80 transition-opacity">
                    <img src="{{ asset('assets/logo/logo-horizontal.png') }}" alt="ROODIO Logo" class="w-full h-auto">
                </a>

                <!-- CTA Button -->
                <div>
                    <x-button behaviour="navigation" navLink="auth/login" content="Let's Play Music!" customClass="!bg-white !text-primary-85 hover:!bg-primary-10 font-secondaryAndButton font-bold text-xs md:text-sm tracking-wider uppercase !py-2 md:!py-3 !px-4 md:!px-6 rounded-none skew-x-[-10deg] transition-all hover:skew-x-0 shadow-[2px_2px_0px_0px_rgba(255,255,255,0.3)] md:shadow-[4px_4px_0px_0px_rgba(255,255,255,0.3)] hover:shadow-none" style="zoom:0.75;"></x-button>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main>

            <!-- Hero Section -->
            <section class="relative min-h-[100vh] flex items-center bg-primary-85 text-white overflow-hidden pb-10 md:pb-0">
                <!-- Full Bg Pattern (Subtle) -->
                <div class="absolute inset-0 bg-[radial-gradient(#ffffff_1px,transparent_1px)] [background-size:60px_60px] opacity-[0.03] pointer-events-none"></div>
                
                <!-- NEW: Specific Graphics (Donuts & Dot Grid Patches) based on reference -->
                <!-- Large Yellow Donut (Top Left) -->
                <div class="absolute top-24 -left-10 w-32 h-32 md:w-48 md:h-48 shape-donut border-[12px] md:border-[16px] border-secondary-happy-50 opacity-20 animate-rotate-scale pointer-events-none" data-aos="zoom-in-down"></div>
                
                <!-- Dot Grid Patch (Top Right) -->
                <div class="absolute top-20 right-10 w-24 h-24 shape-dot-grid text-secondary-happy-50/20 animate-float-medium pointer-events-none"></div>

                <!-- Green Donut (Bottom Right) -->
                <div class="absolute bottom-32 -right-10 w-36 h-36 shape-donut border-[12px] border-secondary-relaxed-50 opacity-10 animate-float-slow pointer-events-none"></div>
                
                <!-- Small Blue Donut (Middle Left) -->
                <div class="absolute top-1/2 left-10 w-16 h-16 shape-donut border-[6px] border-secondary-sad-50 opacity-20 animate-float-fast pointer-events-none"></div>
                
                <!-- Dot Grid Patch (Bottom Left) -->
                <div class="absolute bottom-20 left-20 w-32 h-32 shape-dot-grid text-white/20 animate-float-slow pointer-events-none"></div>

                <!-- NEW: Enrichment "Rame" Elements -->
                <!-- Floating Music Notes -->
                <div class="absolute top-32 left-[20%] text-4xl text-secondary-happy-50/60 animate-float-slow pointer-events-none font-bold drop-shadow-lg" style="animation-duration: 7s;" data-aos="zoom-in">🎵</div>
                <div class="absolute bottom-40 right-[25%] text-5xl text-secondary-sad-50/60 animate-float-medium pointer-events-none font-bold drop-shadow-lg" style="animation-duration: 9s;" data-aos="zoom-in">🎶</div>
                <div class="absolute top-[15%] right-[30%] text-3xl text-secondary-relaxed-50/40 animate-float-fast pointer-events-none font-bold" data-aos="zoom-in">♪</div>

                <!-- Giant Faint Circle Outline -->
                <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[550px] border-[2px] border-dashed border-white/20 rounded-full animate-rotate-scale pointer-events-none"></div>
                <!-- Smaller Inner Circle -->
                <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[350px] h-[350px] border-[2px] border-white/20 rounded-full animate-pulse-glow pointer-events-none"></div>

                <!-- Existing Elements (Retained for density) -->
                <!-- Floating Mood Icons (Scattered) -->
                <!-- These will be replaced/augmented by the large orbs below, but kept for depth if needed (toned down) -->

                <div class="container mx-auto px-4 mb-12 md:px-6 relative z-10 h-full flex flex-col items-center justify-center pt-20 lg:pb-10  min-h-[80vh]">
                    
                    <!-- Center: Typography & CTA -->
                    <div class="text-center max-w-4xl mx-auto relative z-20" data-aos="zoom-in-up">
                        <div class="inline-block px-4 py-1 md:px-6 md:py-2 bg-secondary-happy-85 text-white font-secondaryAndButton text-micro md:text-small font-bold uppercase tracking-widest mb-6 md:mb-8 skew-x-[-10deg] shadow-lg">
                            <span class="block skew-x-[10deg]">Get The Moo-Dies, Listen to The Music</span>
                        </div>
                        <h1 class="font-primary text-5xl md:text-7xl lg:text-8xl font-bold mb-3 drop-shadow-2xl">
                            Listen Music<br>
                            <span class="text-primary-10 relative inline-block">
                                With 
                                <span class='text-white'>R</span><span class='text-secondary-happy-85'>O</span><span class='text-secondary-relaxed-85'>O</span><span class='text-secondary-sad-85'>D</span><span class='text-secondary-angry-85'>I</span><span class='text-white'>O</span>.
                                <hr class='border-2 border-secondary-happy-50 mt-4'>
                            </span>
                        </h1>
                        <p class="font-secondaryAndButton text-white text-small md:text-body-size lg:text-paragraph mb-8 md:mb-12 leading-relaxed max-w-2xl mx-auto drop-shadow-md">
                            Roodio matches music to your mood.
                            <br>Enjoy songs that fit how you feel right now.
                        </p>
                        
                        <div class="flex justify-center gap-6">
                            <x-button behaviour="navigation" navLink="auth/login" content="Listen the song now!" mood="relaxed" customClass="!bg-secondary-happy-85 !text-white hover:!bg-secondary-happy-70 font-bold !px-8 md:!px-12 !py-4 md:!py-5 text-base md:text-lg shadow-[4px_4px_0px_0px_rgba(0,0,0,0.3)] md:shadow-[8px_8px_0px_0px_rgba(0,0,0,0.3)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,0.3)] transition-all transform hover:translate-x-1 hover:translate-y-1 rounded-none" style="zoom:0.95;"></x-button>
                        </div>
                    </div>

                    <!-- Scattered Large Glass Orbs (Surrounding the Text) -->
                    
                    <!-- Orb 1: Happy (Top Left) -->
                    <div class="absolute top-[10%] md:top-[12%] left-[2%] md:left-[5%] w-32 h-32 md:w-56 md:h-56 flex items-center justify-center animate-float-slow group cursor-pointer hover:scale-110 transition-transform duration-500 z-10">
                        <div class="relative w-24 h-24 md:w-40 md:h-40 flex items-center justify-center" data-aos="fade-down-right">
                            <img src="{{ asset('assets/moods/happy.png') }}" class="w-full h-full object-contain drop-shadow-2xl transform group-hover:rotate-12 transition-transform duration-500"  alt="Happy">
                        </div>
                        <!-- Tooltip -->
                        <div class="absolute -bottom-2 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-2 group-hover:translate-y-0 pointer-events-none">
                            <div class="bg-secondary-happy-85 text-white text-xs md:text-sm font-bold px-4 py-1.5 rounded-full shadow-lg tracking-wider relative">
                                <span class="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-secondary-happy-85 rotate-45"></span>
                                HAPPY
                            </div>
                        </div>
                    </div>

                    <!-- Orb 2: Relaxed (Top Right) -->
                    <div class="absolute top-[10%] md:top-[15%] right-[2%] md:right-[5%] w-28 h-28 md:w-48 md:h-48 flex items-center justify-center animate-float-medium group cursor-pointer hover:scale-110 transition-transform duration-500 z-10" style="animation-delay: 1.5s;">
                            <div class="relative w-20 h-20 md:w-36 md:h-36 flex items-center justify-center" data-aos="fade-down-left">
                            <img src="{{ asset('assets/moods/relaxed.png') }}" class="w-full h-full object-contain drop-shadow-2xl transform group-hover:-rotate-12 transition-transform duration-500" alt="Relaxed">
                        </div>
                        <!-- Tooltip -->
                        <div class="absolute -bottom-2 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-2 group-hover:translate-y-0 pointer-events-none">
                            <div class="bg-secondary-relaxed-85 text-white text-xs md:text-sm font-bold px-4 py-1.5 rounded-full shadow-lg tracking-wider relative">
                                <span class="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-secondary-relaxed-85 rotate-45"></span>
                                RELAXED
                            </div>
                        </div>
                    </div>

                    <!-- Orb 3: Sad (Bottom Left) -->
                    <div class="absolute bottom-[5%] md:bottom-[10%] left-[5%] md:left-[10%] w-36 h-36 md:w-60 md:h-60 flex items-center justify-center animate-float-slow group cursor-pointer hover:scale-110 transition-transform duration-500 z-10" style="animation-delay: 0.5s;">
                        <div class="relative w-28 h-28 md:w-44 md:h-44 flex items-center justify-center" data-aos="fade-up-right">
                            <img src="{{ asset('assets/moods/sad.png') }}" class="w-full h-full object-contain drop-shadow-2xl transform group-hover:scale-105 transition-transform duration-500" alt="Sad">
                        </div>
                        <!-- Tooltip -->
                        <div class="absolute -bottom-2 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-2 group-hover:translate-y-0 pointer-events-none">
                            <div class="bg-secondary-sad-85 text-white text-xs md:text-sm font-bold px-4 py-1.5 rounded-full shadow-lg tracking-wider relative">
                                <span class="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-secondary-sad-85 rotate-45"></span>
                                SAD
                            </div>
                        </div>
                    </div>

                    <!-- Orb 4: Angry (Bottom Right) -->
                    <div class="absolute bottom-[8%] md:bottom-[12%] right-[5%] md:right-[12%] w-32 h-32 md:w-52 md:h-52 flex items-center justify-center animate-float-fast group cursor-pointer hover:scale-110 transition-transform duration-500 z-10" style="animation-delay: 2s;">
                        <div class="relative w-24 h-24 md:w-36 md:h-36 flex items-center justify-center" data-aos="fade-up-left">
                            <img src="{{ asset('assets/moods/angry.png') }}" class="w-full h-full object-contain drop-shadow-2xl transform group-hover:shake transition-transform duration-500" alt="Angry">
                        </div>
                        <!-- Tooltip -->
                        <div class="absolute -bottom-2 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-2 group-hover:translate-y-0 pointer-events-none">
                            <div class="bg-secondary-angry-85 text-white text-xs md:text-sm font-bold px-4 py-1.5 rounded-full shadow-lg tracking-wider relative">
                                <span class="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-secondary-angry-85 rotate-45"></span>
                                ANGRY
                            </div>
                        </div>
                    </div>

                </div>
                </div>

                <!-- Wave Divider Bottom (New Path) -->
                <div class="wave-bottom">
                    <svg data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none">
                        <path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V120H0Z" class="shape-fill"></path>
                    </svg>
                </div>
            </section>

            <!-- Features Section (Carousel) -->
            <section id="features" class="py-16 md:py-24 bg-gray-100 relative overflow-hidden">
                <!-- Decorative Shapes (Density) -->
                <div class="absolute top-0 right-0 w-48 h-48 md:w-64 md:h-64 bg-secondary-happy-10 rounded-full opacity-50 blur-3xl translate-x-1/2 -translate-y-1/2"></div>
                <div class="absolute bottom-0 left-0 w-48 h-48 md:w-64 md:h-64 bg-secondary-relaxed-10 rounded-full opacity-50 blur-3xl -translate-x-1/2 translate-y-1/2"></div>
                
                <!-- NEW: Specific Graphics -->
                <div class="absolute top-[10%] left-[5%] w-24 h-24 shape-donut border-[12px] border-secondary-sad-50 opacity-40 animate-float-medium pointer-events-none"></div>
                <div class="absolute bottom-[20%] right-[10%] w-32 h-32 shape-dot-grid text-black/20 animate-float-slow pointer-events-none"></div>
                
                <!-- Extra Ghost Icons (Rame) - Diversified -->
                <div class="absolute top-20 right-[15%] opacity-20 animate-float-slow pointer-events-none transform rotate-12">
                    <img src="{{ asset('assets/moods/happy.png') }}" class="w-32 h-32 grayscale brightness-90" alt="">
                </div>
                <div class="absolute bottom-10 left-[10%] opacity-20 animate-float-medium pointer-events-none transform -rotate-12">
                    <img src="{{ asset('assets/moods/angry.png') }}" class="w-40 h-40 grayscale brightness-90" alt="">
                </div>
                <!-- More Scattered Icons -->
                <div class="absolute top-[40%] left-[2%] opacity-15 animate-float-fast pointer-events-none w-20 h-20 transform rotate-45">
                    <img src="{{ asset('assets/moods/relaxed.png') }}" class="w-full h-full grayscale brightness-90" alt="">
                </div>
                <div class="absolute bottom-[30%] right-[5%] opacity-15 animate-float-slow pointer-events-none w-24 h-24 transform -rotate-12">
                    <img src="{{ asset('assets/moods/sad.png') }}" class="w-full h-full grayscale brightness-90" alt="">
                </div>
                <!-- Geometric Squiggles (CSS Shapes) -->
                <div class="absolute top-[15%] left-[40%] w-0 h-0 border-l-[15px] border-l-transparent border-r-[15px] border-r-transparent border-b-[26px] border-b-secondary-happy-50/30 transform rotate-12 animate-float-medium pointer-events-none"></div>
                <div class="absolute bottom-[15%] right-[40%] w-8 h-8 border-4 border-secondary-sad-50/30 transform rotate-45 animate-spin-slow pointer-events-none"></div>

                <div class="container mx-auto px-4 md:px-6 relative z-10">
                    <div class="text-center max-w-2xl mx-auto mb-10 md:mb-16" data-aos="fade-up">
                        <h2 class="font-primary text-paragraph md:text-subtitle lg:text-title font-bold text-primary-50 mb-3 md:mb-4">WHY MUST ROODIO?</h2>
                        <div class="w-16 md:w-24 h-1 bg-secondary-happy-85 mx-auto rounded-full"></div>
                        <p class="font-secondaryAndButton text-secondary-angry-100 mt-4 md:mt-6 text-small md:text-body-size px-4">Discover features made to fit your daily life. <br>Swipe to see what Roodio can do for you.</p>
                    </div>

                    <!-- Swiper - Menambah lebih banyak fitur -->
                    <div class="swiper featuresSwiper px-4 pb-16" data-aos="fade-up" data-aos-delay="200">
                        <div class="swiper-wrapper">
                            @foreach ($features as $feature)
                                <div class="swiper-slide h-auto">
                                    <div class="h-full group bg-white p-6 md:p-8 shadow-md hover:shadow-xl transition-all duration-300 
                                        border-t-4 border-secondary-{{ $feature['color'] }}-85 
                                        relative overflow-hidden rounded-xl cursor-grab active:cursor-grabbing">

                                        <div class="absolute top-0 right-0 w-20 h-20 md:w-24 md:h-24 
                                            bg-secondary-{{ $feature['color'] }}-10 
                                            rounded-bl-[80px] md:rounded-bl-[100px] -mr-4 -mt-4 
                                            transition-transform group-hover:scale-110">
                                        </div>

                                        <div class="relative z-10">

                                            <div class="w-12 h-12 md:w-14 md:h-14 
                                                bg-secondary-{{ $feature['color'] }}-85 
                                                text-white flex items-center justify-center 
                                                text-xl md:text-2xl mb-4 md:mb-6 shadow-md 
                                                skew-x-[-10deg] rounded-sm 
                                                transform group-hover:rotate-6 transition-transform">

                                                <span class="block skew-x-[10deg]">
                                                    {{ $feature['icon'] }}
                                                </span>
                                            </div>

                                            <h3 class="font-primary text-body-size md:text-paragraph font-bold 
                                                text-secondary-{{ $feature['color'] }}-85 
                                                mb-2 md:mb-3">
                                                {{ $feature['title'] }}
                                            </h3>

                                            <p class="font-secondaryAndButton text-small md:text-body-size 
                                                text-primary-60
                                                leading-relaxed">
                                                {{ $feature['desc'] }}
                                            </p>

                                        </div>
                                    </div>
                                </div>
                            @endforeach
                        </div>                        
                        <div class="swiper-button-next text-primary-85 opacity-0 md:opacity-100 transition-opacity"></div>
                        <div class="swiper-button-prev text-primary-85 opacity-0 md:opacity-100 transition-opacity"></div>
                    </div>
                    <div class="swiper-pagination"></div>
                </div>
            </section>

            <!-- Team Section -->
            <section id="team" class="py-16 md:py-24 bg-primary-100 text-white relative overflow-hidden">
                <!-- Dot Pattern Background -->
                <div class="absolute inset-0 bg-pattern-dots pointer-events-none opacity-10"></div>
                
                <!-- NEW: Specific Graphics -->
                <div class="absolute top-20 left-10 w-24 h-24 shape-dot-grid text-white/20 animate-float-medium pointer-events-none"></div>
                <div class="absolute bottom-40 right-10 w-12 h-12 shape-donut border-[7px] border-secondary-happy-50 opacity-20 animate-rotate-scale pointer-events-none"></div>
                
                <!-- Large Gradient Blobs for Depth -->
                <div class="absolute top-0 right-0 w-[500px] h-[500px] bg-secondary-happy-50/20 rounded-full blur-3xl pointer-events-none -translate-y-1/2 translate-x-1/2 mix-blend-overlay"></div>
                <div class="absolute bottom-0 left-0 w-[500px] h-[500px] bg-secondary-sad-50/20 rounded-full blur-3xl pointer-events-none translate-y-1/2 -translate-x-1/2 mix-blend-overlay"></div>

                <!-- Extra Ghost Icons (Rame) - Increased Density -->
                <div class="absolute top-[30%] right-[5%] opacity-20 animate-float-fast pointer-events-none">
                    <img src="{{ asset('assets/moods/relaxed.png') }}" class="w-28 h-28" alt="">
                </div>
                <div class="absolute bottom-[10%] left-[20%] opacity-20 animate-float-slow pointer-events-none">
                    <img src="{{ asset('assets/moods/sad.png') }}" class="w-48 h-48" alt="">
                </div>
                <!-- Additional Team BG Icons -->
                <div class="absolute top-[15%] left-[15%] opacity-15 animate-float-medium pointer-events-none transform rotate-180">
                    <img src="{{ asset('assets/moods/happy.png') }}" class="w-32 h-32" alt="">
                </div>
                <div class="absolute top-[65%] right-[15%] opacity-15 animate-float-medium pointer-events-none transform -rotate-30">
                    <img src="{{ asset('assets/moods/angry.png') }}" class="w-42 h-42" alt="">
                </div>
                
                <div class="absolute top-[10%] left-[40%] w-16 h-16 shape-donut border-[4px] border-white/30 animate-float-medium pointer-events-none"></div>
                <div class="absolute bottom-[20%] right-[30%] w-12 h-12 shape-donut border-[4px] border-white/20 animate-float-slow pointer-events-none"></div>
                
                <!-- Wave Divider Top -->
                <div class="wave-top">
                    <svg data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none">
                        <path d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V0H0V27.35A600.21,600.21,0,0,0,321.39,56.44Z" class="shape-fill"></path>
                    </svg>
                </div>

                <div class="container mx-auto px-6 relative z-10 pt-10 md:pt-16">
                    <div class="text-center mb-16 md:mb-20" data-aos="fade-up">
                         <h2 class="font-primary text-subtitle md:text-title font-bold mb-4">Meet the Crews</h2>
                         <p class="font-secondaryAndButton text-white max-w-2xl mx-auto text-micro md:text-small">The team behind the development and evaluation of ROODIO.</p>
                    </div>

                    <!-- Developers Group -->
                    <div class="mb-16 md:mb-24">
                        <div class="flex items-center gap-4 mb-10 md:mb-12 justify-center" data-aos="fade-up">
                            <h3 class="font-primary text-paragraph md:text-subtitle font-bold text-secondary-happy-60 uppercase tracking-wider">Development Team</h3>
                        </div>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-10 md:gap-14 max-w-7xl mx-auto">
                            @foreach($developers as $dev)

                                @php
                                    $photo = isset($dev['photo'])
                                        ? asset('assets/teams/developers/' . $dev['photo'])
                                        : asset('assets/defaults/user.jpg');
                                @endphp

                                <div class="flex flex-col items-center group"
                                    data-aos="fade-left"
                                    data-aos-delay="{{ $loop->index * 100 }}">

                                    <!-- Circle Image -->
                                    <div class="w-52 h-52 md:w-48 md:h-48 rounded-full bg-white flex items-center justify-center overflow-hidden mb-4 md:mb-6 shadow-xl border-2 border-white group-hover:scale-105 transition-transform duration-300 relative z-10">

                                        <img src="{{ $photo }}"
                                            alt="{{ $dev['name'] }}"
                                            class="w-full h-full object-cover">
                                    </div>

                                    <h4 class="font-secondaryAndButton text-center text-small md:text-body-size text-secondary-happy-60 group-hover:text-white transition-colors">
                                        {{ $dev['name'] }}
                                    </h4>

                                    <p class="font-secondaryAndButton text-center text-micro md:text-small text-white mt-1">
                                        {{ $dev['role'] }}
                                    </p>

                                </div>

                                @endforeach

                        </div>
                    </div>

                    <!-- Academic Advisors -->
                    <div>
                        <div class="flex items-center gap-4 mb-10 md:mb-12 justify-center" data-aos="fade-up">
                            <h3 class="font-primary text-paragraph md:text-subtitle font-bold text-secondary-relaxed-60 uppercase tracking-wider">Academic Advisors</h3>
                        </div>

                        <div class="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-3xl mx-auto">
                            @foreach($advisors as $advisor)
                                @php
                                    $photo = isset($advisor['photo'])
                                            ? asset('assets/teams/advisors/' . $advisor['photo'])
                                            : asset('assets/defaults/user.jpg');
                                @endphp

                                <div class="flex flex-col items-center group"
                                    data-aos="fade-right"
                                    data-aos-delay="{{ $loop->index * 100 }}">

                                    <!-- Circle Image -->
                                    <div class="w-52 h-52 md:w-48 md:h-48 rounded-full bg-white flex items-center justify-center overflow-hidden mb-4 md:mb-6 shadow-xl border-2 border-white group-hover:scale-105 transition-transform duration-300 relative z-10">
                                        <img src="{{ $photo }}"
                                            alt="{{ $advisor['name'] }}"
                                            class="w-full h-full object-cover">
                                    </div>

                                    <h4 class="font-secondaryAndButton text-center text-small md:text-body-size text-secondary-relaxed-60 group-hover:text-white transition-colors">
                                        {{ $advisor['name'] }}
                                    </h4>

                                    <p class="font-secondaryAndButton text-center text-micro md:text-small text-white mt-1">
                                        {{ $advisor['role'] }}
                                    </p>
                                </div>
                            @endforeach
                        </div>
                    </div>
                </div>
            </section>
        </main>

        <!-- Footer -->
        <footer id="footer" class="bg-primary-100 border-t border-white/10 pt-10 md:pt-16 pb-8 text-white relative z-20">
            <div class="container mx-auto px-6">
                <div class="grid md:grid-cols-4 gap-8 md:gap-12 mb-10 md:mb-16 text-center md:text-left">
                    <!-- Brand -->
                    <div class="col-span-1 md:col-span-2 md:pr-8 flex flex-col items-center md:items-start">
                         <div class="w-24 md:w-32 mb-4 md:mb-6">
                            <img src="{{ asset('assets/logo/logo-horizontal.png') }}" alt="Logo" class="w-full grayscale brightness-200 hover:brightness-100 hover:grayscale-0 transition-all duration-500">
                         </div>
                         <p class="font-secondaryAndButton text-primary-20 text-xs md:text-sm leading-relaxed mb-6 max-w-sm">
                             Pioneering the future of emotional audio intelligence. Roodio connects human emotion with sonic landscapes through data-driven curation.
                         </p>
                         <a href="mailto:roodio.team@gmail.com" class="font-secondaryAndButton text-primary-30 hover:text-white transition-colors text-xs md:text-sm font-medium flex items-center gap-2">
                            <span>📧</span> roodio.team@gmail.com
                         </a>
                    </div>

                    <!-- Links -->
                    <div>
                        <h4 class="font-primary font-bold text-white mb-4 md:mb-6 uppercase text-xs md:text-sm tracking-wider">Platform</h4>
                        <ul class="space-y-3 font-secondaryAndButton text-xs md:text-sm text-primary-30">
                            <li><a href="#" class="hover:text-white transition-colors">Technology</a></li>
                            <li><a href="#" class="hover:text-white transition-colors">Solutions</a></li>
                            <li><a href="#" class="hover:text-white transition-colors">Integration</a></li>
                        </ul>
                    </div>

                    <!-- Legal -->
                    <div>
                        <h4 class="font-primary font-bold text-white mb-4 md:mb-6 uppercase text-xs md:text-sm tracking-wider">Legal</h4>
                        <ul class="space-y-3 font-secondaryAndButton text-xs md:text-sm text-primary-30">
                            <li><a href="#" class="hover:text-white transition-colors">Privacy Notice</a></li>
                            <li><a href="#" class="hover:text-white transition-colors">Terms of Use</a></li>
                            <li><a href="#" class="hover:text-white transition-colors">Accessibility</a></li>
                        </ul>
                    </div>
                </div>

                <div class="border-t border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
                    <p class="font-secondaryAndButton text-primary-40 text-xs text-center md:text-left">© {{ date('Y') }} PT ROODIO Indonesia. All rights reserved.</p>
                </div>
            </div>
        </footer>

    </div>
@endsection