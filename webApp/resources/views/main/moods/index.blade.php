@extends('layouts.main')


@section('title', 'ROODIO - Moods')


@push('style')
    <link rel="stylesheet" href="https://unpkg.com/tippy.js@6/dist/tippy.css"/>
@endpush


@push('script')
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="{{ asset('js/pages/main/mood.js') }}" defer></script>
    <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.20/index.global.min.js'></script>
    <script src="https://unpkg.com/@popperjs/core@2"></script>
    <script src="https://unpkg.com/tippy.js@6"></script>
    <script src="https://cdn.jsdelivr.net/npm/matter-js@0.19.0/build/matter.min.js"></script>
@endpush


@php
    // dd($weekly, $monthly, $yearly);
    // dd($weekly[0]->mood->type);
    $startDate = $weekly[0]['startDate']; 
    $endDate = $weekly[0]['endDate'];

    $calendarData = $monthly->map(function ($item) {
    return [
        'title' => $item['type'],
        'start' => $item['date'],   
        'total' => $item['total'],
        'type'  => $item['type'],
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
@endphp


<script>
    window.moodWeeklyData = @json(collect($weekly)->map(fn($item) => [
        'type' => $item['type'],
        'total' => (int)$item['total']
    ])->toArray());
    
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
</script>


@section('mainContent')
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
    <form action="{{ route('threads.index') }}" method="GET">
        <div class='mb-7 flex flex-row gap-3 w-full lg:justify-end contentFadeLoad'>
            <x-filterButton id='weekly' name='filter' value='weekly' :mood='$mood' label='Weekly' onchange="this.form.submit()"></x-filterButton>
            <x-filterButton id='monthly' name='filter' value='monthly' :mood='$mood' label='Monthly' onchange="this.form.submit()"></x-filterButton>
            <x-filterButton id='yearly' name='filter' value='yearly' :mood='$mood' label='Yearly' onchange="this.form.submit()"></x-filterButton>
        </div>
    </form>

    <div id='weeklyMood' class='contentFadeLoad'>
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

    {{-- <div id="calendar"></div> --}}
    {{-- <div id="bubble-container" style="height:600px;"></div> --}}
    {{-- <div id="bubble-container" style="width:90%; height:600px;"></div> --}}

    <script>


 document.addEventListener('DOMContentLoaded', function () {

    var calendar = new FullCalendar.Calendar(
        document.getElementById('calendar'),
        {
            initialView: 'dayGridMonth',
            height: 650,
            events: @json($calendarData),

            eventDidMount: function(info) {

                tippy(info.el, {
                    content: `
                        <strong>${info.event.title}</strong><br>
                        Total: ${info.event.extendedProps.total}
                    `,
                    allowHTML: true,
                    theme: 'light-border',
                    animation: 'scale',
                });

            }
        }
    );

    calendar.render();
});

document.addEventListener("DOMContentLoaded", function () {

    const { Engine, Render, Runner, Bodies, World, Mouse, MouseConstraint, Events, Body } = Matter;

    const container = document.getElementById('bubble-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    const engine = Engine.create();
    const world = engine.world;
    engine.gravity.y = 1;

    const render = Render.create({
        element: container,
        engine: engine,
        options: {
            width: width,
            height: height,
            wireframes: false,
            background: '#0b0f2a'
        }
    });

    Render.run(render);
    Runner.run(Runner.create(), engine);

    // ===== WALLS =====
    World.add(world, [
        Bodies.rectangle(width/2, height+30, width, 60, { isStatic: true }),
        Bodies.rectangle(-30, height/2, 60, height, { isStatic: true }),
        Bodies.rectangle(width+30, height/2, 60, height, { isStatic: true })
    ]);

    // ===== DRAG MOUSE =====
    const mouse = Mouse.create(render.canvas);
    const mouseConstraint = MouseConstraint.create(engine, {
        mouse: mouse,
        constraint: { stiffness: 0.2, render: { visible: false } }
    });

    World.add(world, mouseConstraint);
    render.mouse = mouse;

    // ===== DATA DARI LARAVEL =====
    const yearlyData = @json($yearly);

    const moodImages = {
        happy: "{{ asset('assets/moods/icons/happy.png') }}",
        sad: "{{ asset('assets/moods/icons/sad.png') }}",
        relaxed: "{{ asset('assets/moods/icons/relaxed.png') }}"
    };

    const textureSize = 512; // ukuran asli PNG kamu
    let moodBodies = [];

    // 🎈 BACKGROUND BALLS
    for (let i = 0; i < 35; i++) {
        const ball = Bodies.circle(
            Math.random() * width,
            Math.random() * -600,
            20 + Math.random() * 25,
            {
                restitution: 0.9,
                frictionAir: 0.01,
                render: {
                    fillStyle: `hsl(${Math.random()*360},70%,60%)`
                }
            }
        );
        World.add(world, ball);
    }

    // 😀 MOOD BALLS
    yearlyData.forEach(item => {

        const radius = 45;
        const diameter = radius * 2;

        const ball = Bodies.circle(
            Math.random() * width,
            -100,
            radius,
            {
                restitution: 0.9,
                frictionAir: 0.01,
                render: {
                    sprite: {
                        texture: moodImages[item.type] ?? '',
                        xScale: 0.05,
                        yScale: 0.05
                    }
                }
            }
        );

        ball.moodData = item;
        moodBodies.push(ball);
        World.add(world, ball);
    });

    // ===== BALIKIN KALO KELUAR AREA =====
    Events.on(engine, "afterUpdate", function() {

        moodBodies.forEach(body => {

            if (body.position.y > height + 200) {
                Body.setPosition(body, { x: Math.random()*width, y: -100 });
                Body.setVelocity(body, { x: 0, y: 0 });
            }

            if (body.position.x < -200 || body.position.x > width + 200) {
                Body.setPosition(body, { x: Math.random()*width, y: -100 });
                Body.setVelocity(body, { x: 0, y: 0 });
            }

        });

    });

    // ===== TOOLTIP STABLE VERSION =====
    let tooltip = null;

    render.canvas.addEventListener('mousemove', function(event) {

        const rect = render.canvas.getBoundingClientRect();
        const mousePosition = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };

        let hoveredBody = null;

        moodBodies.forEach(body => {

            const dx = body.position.x - mousePosition.x;
            const dy = body.position.y - mousePosition.y;
            const distance = Math.sqrt(dx*dx + dy*dy);

            if (distance < body.circleRadius) {
                hoveredBody = body;
            }

        });

        if (hoveredBody) {

            const data = hoveredBody.moodData;

            if (!tooltip) {

                tooltip = tippy(render.canvas, {
                    content: `
                        <strong>${data.month}</strong><br>
                        Mood: ${data.type}<br>
                        Total: ${data.total}
                    `,
                    allowHTML: true,
                    trigger: 'manual',
                    followCursor: true,
                    placement: 'top'
                });

                tooltip.show();
            }

        } else {

            if (tooltip) {
                tooltip.destroy();
                tooltip = null;
            }

        }

    });

});

    </script>
@endsection