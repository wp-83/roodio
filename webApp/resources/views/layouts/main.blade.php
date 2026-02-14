@extends('layouts.master')


@section('title', 'Roodio - Get The MOO-DIES, Listen The Music')




@push('script')
    <script src="{{ asset('js/pages/main/navigation.js?v=2') }}" defer></script>
@endpush


@section('bodyClass', 'max-h-screen h-screen flex flex-col')


@section('bodyContent')
    @yield('overlayContent')
    
    {{-- Top Loading Progress Bar --}}
    @php
        $loadingInfos = [
            'happy' => 'bg-secondary-happy-50',
            'sad' => 'bg-secondary-sad-50',
            'relaxed' => 'bg-secondary-relaxed-50',
            'angry' => 'bg-secondary-angry-50'
        ];
        $loadingColor = $loadingInfos[$mood] ?? 'bg-primary-50';
    @endphp
    
    <div id="top-loading-bar" class="fixed top-0 left-0 w-full h-0.5 z-[9999] hidden">
        <div class="h-full {{ $loadingColor }} transition-all duration-300 ease-out" style="width: 0%" id="top-loading-progress"></div>
    </div>

    <script>
        function showLoadingBar() {
            const bar = document.getElementById('top-loading-bar');
            const progress = document.getElementById('top-loading-progress');
            if(bar && progress) {
                bar.classList.remove('hidden');
                // Small delay to ensure transition works if it was just hidden
                requestAnimationFrame(() => {
                    progress.style.width = '30%';
                });
                
                // Auto increment slightly to show activity
                if (!window.loadingTimer) {
                    window.loadingTimer = setTimeout(() => { 
                        if(progress.style.width === '30%') progress.style.width = '70%'; 
                    }, 500);
                }
            }
        }

        function hideLoadingBar() {
            const bar = document.getElementById('top-loading-bar');
            const progress = document.getElementById('top-loading-progress');
            if(bar && progress) {
                progress.style.width = '100%';
                if (window.loadingTimer) {
                     clearTimeout(window.loadingTimer);
                     window.loadingTimer = null;
                }
                setTimeout(() => {
                    bar.classList.add('hidden');
                    progress.style.width = '0%';
                }, 400); 
            }
        }

        // Handle page navigation (SPA)
        document.addEventListener('livewire:navigating', showLoadingBar);
        document.addEventListener('livewire:navigated', hideLoadingBar);

        // Handle component updates (Filters, Forms)
        document.addEventListener('livewire:init', () => {
             Livewire.hook('request', ({ fail, succeed }) => {
                showLoadingBar();
                succeed(() => hideLoadingBar());
                fail(() => hideLoadingBar());
            });
        });
    </script>

    <div class="shrink-0">
        <x-navbar></x-navbar>
    </div>
    <div class='flex flex-col flex-1 min-h-0'>
        <div class='flex flex-row flex-1 min-h-0 relative'>
            <x-sidebar :mood='$mood' class='relative z-10'></x-sidebar>
            <div id='scrollContainer' class='w-full bg-primary-100 overflow-x-hidden overflow-y-auto min-h-0 p-10 scrollbar scrollbar-thumb-primary-10/75 scrollbar-track-transparent @yield('mainContentContainerClass')'>
                @yield('mainContent')
            </div>
        </div>
    </div>
    @persist('player')
        <x-audioPlayer :mood='$mood'></x-audioPlayer>
    @endpersist
@endsection

@push('script')
    <script src="{{ asset('js/pages/main/audioControl.js?v=2') }}" defer></script>
@endpush
