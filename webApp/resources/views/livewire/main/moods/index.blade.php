<div>
    @push('head')
        <meta name="csrf-token" content="{{ csrf_token() }}">
    @endpush

    @php
        // Prepare data for JS
        $startDate = $weekly->isNotEmpty() ? $weekly[0]['startDate'] : now()->startOfWeek()->format('F jS, Y'); 
        $endDate = $weekly->isNotEmpty() ? $weekly[0]['endDate'] : now()->endOfWeek()->format('F jS, Y');

        $firstMonth = $yearly->first();  
        $lastMonth = $yearly->last();           
        $startMonth = $firstMonth['bulan'] ?? 'January';
        $endMonth = $lastMonth['bulan'] ?? 'December';
        $yearMonth = date('Y');         

        $calendarData = $monthly->map(function ($item) {
            return [
                'title' => $item['type'],
                'start' => $item['date'],   
                'total' => $item['total'],
                'type'  => $item['type'],
            ];
        });

        $yearlyData = $yearly->map(function($item) {
            return [
                'mood' => $item['type'] ?? $item['mood'] ?? 'happy',
                'bulan' => $item['bulan'] ?? $item['month'] ?? 'Unknown',
                'total' => $item['total'] ?? 0,
                'persentase' => $item['persentase'] ?? '0%'
            ];
        });

        $textColor = [
            'happy' => 'text-secondary-happy-30',
            'sad' => 'text-secondary-sad-30',
            'relaxed' => 'text-secondary-relaxed-30',
            'angry' => 'text-secondary-angry-30'
        ];

        $textColorTitleChart = [
            'happy' => 'text-secondary-happy-50',
            'sad' => 'text-secondary-sad-50',
            'relaxed' => 'text-secondary-relaxed-50',
            'angry' => 'text-secondary-angry-50'
        ];

        $borderMood = [
            'happy' => 'border-secondary-happy-70',
            'sad' => 'border-secondary-sad-70',
            'relaxed' => 'border-secondary-relaxed-70',
            'angry' => 'border-secondary-angry-70'
        ];
    @endphp

    <script>
        // Update window data when Livewire updates
        // Note: Since charts usually render once, we might need to re-trigger rendering logic
        // We can hook into Livewire events in the main script file or here.
        
        window.updateMoodData = function() {
             window.moodWeeklyData = @json(collect($weekly)->map(fn($item) => [
                'type' => $item['type'],
                'total' => (int)$item['total']
            ])->toArray());
            
            window.calendarData = @json($calendarData->values()->toArray());
            window.moodYearlyData = @json($yearlyData);
        }
        window.updateMoodData();

        window.moodIcons = {
            happy: "{{ asset('assets/moods/icons/happy.png') }}",
            sad: "{{ asset('assets/moods/icons/sad.png') }}",
            relaxed: "{{ asset('assets/moods/icons/relaxed.png') }}",
            angry: "{{ asset('assets/moods/icons/angry.png') }}"
        };

        window.moodColors = {
            default: {
                happy: '#FF8E2B',
                sad: '#6A4FBF',
                relaxed: '#28C76F',
                angry: '#E63946'
            },

            hover: {
                happy: '#FFB775',
                sad: '#A38FDF',
                relaxed: '#78DAA3',
                angry: '#F0858A'
            }
        };

        window.baseUrl = "{{ asset('') }}";
    </script>


    <div class='mb-5 md:mb-9 w-full'>
        <div class='flex flex-row gap-3'>
            <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="{{ $mood }}" class='h-26 w-26 md:h-32 md:w-32 lg:h-40 lg:w-40'>
            <div class='flex flex-col'>
                <div class='w-0 relative overflow-hidden typingTextAnimation max-w-max '>
                    <p class='font-primary text-title font-bold {{ $textColor[$mood] }} md:text-hero' >Moods</p>
                </div>
                <p class='font-secondaryAndButton text-white text-justify contentFadeLoad text-small md:text-body-size'>Hi, {{ Str::before($user->userDetail->fullname, ' ') }}! Let's see your mood recap!</p>
            </div>
        </div>
    </div>
    
    {{-- Filter Buttons --}}
    <div class='mb-7 flex flex-row gap-3 w-full lg:justify-end contentFadeLoad'>
        <div wire:click="$set('filter', 'weekly')" class="cursor-pointer">
            <x-filterButton id='weekly' name='filter' value='weekly' :mood='$mood' label='Weekly' :active="$filter === 'weekly'"></x-filterButton>
        </div>
        <div wire:click="$set('filter', 'monthly')" class="cursor-pointer">
            <x-filterButton id='monthly' name='filter' value='monthly' :mood='$mood' label='Monthly' :active="$filter === 'monthly'"></x-filterButton>
        </div>
        <div wire:click="$set('filter', 'yearly')" class="cursor-pointer">
            <x-filterButton id='yearly' name='filter' value='yearly' :mood='$mood' label='Yearly' :active="$filter === 'yearly'"></x-filterButton>
        </div>
    </div>

    {{-- Weekly Mood --}}
    <div id='weeklyMood' class='contentFadeLoad {{ $filter === "weekly" ? "" : "hidden" }}'>
        <div class='w-full flex flex-col items-center mb-10'>
            <p class='font-primary text-body-size md:text-paragraph lg:text-subtitle font-bold {{ $textColorTitleChart[$mood] }}'>WEEKLY MOOD RECAP</p>
            <p class='font-secondaryAndButton text-white text-micro md:text-body-size'>{{ $startDate }} - {{ $endDate }}</p>
        </div>
        <div class='w-full' style='height: 
            @php 
                $count = count($weekly); 
                if ($count == 0){
                    echo 0 . 'px';
                } else if ($count <= 2) {
                    echo $count * 115 . 'px';
                } else {
                    echo '375px';
                }
            @endphp
        ;'>
            <canvas id='moodChart' class='w-full h-full'></canvas>
        </div>
    </div>

    {{-- Monthly Mood --}}
    <div id='monthlyMood' class='contentFadeLoad {{ $filter === "monthly" ? "" : "hidden" }}'>
        <p class='w-full flex justify-center font-primary text-body-size md:text-paragraph lg:text-subtitle font-bold {{ $textColorTitleChart[$mood] }}'>MONTHLY MOOD RECAP</p>
        <div id='timeCalendar' class='w-full flex flex-row items-center justify-center gap-2 mb-7 font-secondaryAndButton text-white text-micro md:text-body-size'>
            <p id='month'></p>
            <p id='year'></p>
        </div>
        <div id='calendarSummary' class='font-secondaryAndButton text-white bg-white/20 w-max p-3 rounded-md mb-6'>
            <p class='text-sma;; lg:text-body-size font-bold {{ $textColorTitleChart[$mood] }}'>This Month Dominant</p>
            <div class='flex flex-row gap-2 items-center'>
                <img src="" alt="dominantMood" id='dominantMoodImage' class='w-16 h-16'>
                <div>
                    <p id='moodDominant' class='text-small lg:text-body-size font-bold'></p>
                    <div class='flex flex-row gap-2 text-micro'>
                        <p id='percentageMoodDominant'></p>
                        <p>|</p>
                        <div class='flex flex-row gap-1'>
                            <p id='totalMoodDominant'></p>
                            <p>day(s)</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div id="calendar"></div>
    </div>

    {{-- Yearly Mood --}}
    <div id='yearlyMood' class='contentFadeLoad {{ $filter === "yearly" ? "" : "hidden" }}'>
        <div class='w-full flex flex-col items-center mb-10'>
            <p class='font-primary text-body-size md:text-paragraph lg:text-subtitle font-bold {{ $textColorTitleChart[$mood] }}'>YEARLY MOOD RECAP</p>
            <p class='font-secondaryAndButton text-white text-micro md:text-body-size'>{{ $startMonth . ' ' . $yearMonth }} - {{ $endMonth . ' ' . $yearMonth }}</p>
        </div>
        <div id="moodYear" class="w-full h-[70vh] sm:h-[60vh] md:h-[50vh] lg:h-[60vh] min-h-[320px] max-h-[650px] border-3 {{ $borderMood[$mood] }} rounded-lg overflow-hidden border-t-0"></div>
    </div>
    <script>
        document.addEventListener('livewire:initialized', () => {
            Livewire.on('mood-filter-updated', () => {
                setTimeout(() => {
                    if (window.initAllMoodComponents) {
                        window.initAllMoodComponents();
                    }
                }, 100); 
            });
        });
    </script>
</div>
