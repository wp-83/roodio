@props([
    'id',
    'value',
    'mood',
    'label'
])


@php
    $moodStyle = [
        'happy' => 'bg-secondary-happy-50/35 group-hover:bg-secondary-happy-20',
        'sad' => 'bg-secondary-sad-50/35 group-hover:bg-secondary-sad-20',
        'relaxed' => 'bg-secondary-relaxed-50/35 group-hover:bg-secondary-relaxed-20',
        'angry' => 'bg-secondary-angry-50/35 group-hover:bg-secondary-angry-20'
    ];

    $textStyle = [
        'happy' => 'group-hover:text-secondary-happy-100',
        'sad' => 'group-hover:text-secondary-sad-100',
        'relaxed' => 'group-hover:text-secondary-relaxed-100',
        'angry' => 'group-hover:text-secondary-angry-100'
    ];
@endphp


<button class='cursor-pointer group' id='{{ $id }}' value='{{ $value }}'>
    <div {{ $attributes->merge([
        "class" => 'font-secondaryAndButton text-small py-1 px-2 rounded-md md:px-3 md:text-body-size duration-100 ' . $moodStyle[$mood] . ' '
    ]) }}>
        <p class='text-white group-hover:font-bold {{ $textStyle[$mood] }}'>{{ $label }}</p>
    </div>
</button>