@props([
    'mood',
    'id',
    'name',
    'value',
    'label',
    'active' => null
])


@php
    $moodStyle = [
        'happy' => 'bg-secondary-happy-50/35',
        'sad' => 'bg-secondary-sad-50/35',
        'relaxed' => 'bg-secondary-relaxed-50/35',
        'angry' => 'bg-secondary-angry-50/35'
    ];

    $moodHoverStyle = [
        'happy' => 'group-hover:bg-secondary-happy-20',
        'sad' => 'group-hover:bg-secondary-sad-20',
        'relaxed' => 'group-hover:bg-secondary-relaxed-20',
        'angry' => 'group-hover:bg-secondary-angry-20'
    ];

    $textStyle = [
        'happy' => 'group-hover:text-secondary-happy-100',
        'sad' => 'group-hover:text-secondary-sad-100',
        'relaxed' => 'group-hover:text-secondary-relaxed-100',
        'angry' => 'group-hover:text-secondary-angry-100'
    ];

    $activeStyle = [
        'happy' => 'bg-secondary-happy-20',
        'sad' => 'bg-secondary-sad-20',
        'relaxed' => 'bg-secondary-relaxed-20',
        'angry' => 'bg-secondary-angry-20'
    ];

    $textActiveStyle = [
        'happy' => 'text-secondary-happy-100',
        'sad' => 'text-secondary-sad-100',
        'relaxed' => 'text-secondary-relaxed-100',
        'angry' => 'text-secondary-angry-100'
    ];

    if ($active !== null) {
        $isActive = $active;
    } else {
        $activeFilter = request('filter', 'all'); // Default to 'all' if no filter specified
        $isActive = ($activeFilter == $value);
    }
@endphp


<button {{ $attributes->merge(['type' => 'submit']) }} class="{{ ($isActive) ? 'cursor-default' : 'group cursor-pointer' }}" id='{{ $id }}' name='{{ $name }}' value='{{ $value }}' {{ ($isActive) ? 'disabled' : ''}}>
    <div class="font-secondaryAndButton text-small py-1 px-2 rounded-md md:px-3 md:text-body-size duration-100 {{ $moodHoverStyle[$mood] }} {{ ($isActive) ? $activeStyle[$mood] : $moodStyle[$mood] }}">
        <p class="group-hover:font-bold {{ $textStyle[$mood] . ' ' . (($isActive) ? ($textActiveStyle[$mood] . ' font-bold') : 'text-white') }}">{{ $label }}</p>
    </div>
</button>