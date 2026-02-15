{{-- @vite(['resources/css/app.css', 'resources/js/app.js']) --}}

@props([
  'behaviour' => 'action', //action OR navigation
  'actionType' => 'button',
  'navType' => 'button', //button OR text
  'navLink' => null,
  'content' => 'button',
  'mood' => null //happy, sad, relaxed, angry
])

@php
    $navLink ??= route('roodio');

    $baseStyle = 'relative overflow-hidden w-full px-4 py-1 my-1 font-secondaryAndButton rounded-3xl cursor-pointer text-mediumBtn text-center font-bold before:absolute before:inset-0 before:-translate-x-full before:transition-transform before:duration-400 hover:before:translate-x-0 md:text-largeBtn md:px-5 md:py-1.5 disabled:hover:before:-translate-x-full disabled:opacity-70 disabled:cursor-not-allowed disabled:bg-shadedOfGray-30';

    $moodStyle = [
      'happy' => 'bg-secondary-happy-20 text-secondary-happy-100 before:bg-secondary-happy-85 hover:text-secondary-happy-10 disabled:hover:text-secondary-happy-100',
      'sad' => 'bg-secondary-sad-20 text-secondary-sad-100 before:bg-secondary-sad-85 hover:text-secondary-sad-10 disabled:hover:text-secondary-sad-100',
      'relaxed' => 'bg-secondary-relaxed-20 text-secondary-relaxed-100 before:bg-secondary-relaxed-85 hover:text-secondary-relaxed-10 disabled:hover:text-secondary-relaxed-100',
      'angry' => 'bg-secondary-angry-20 text-secondary-angry-100 before:bg-secondary-angry-85 hover:text-secondary-angry-10 disabled:hover:text-secondary-angry-100',
      'grayscale' => 'bg-shadedOfGray-10 text-shadedOfGray-60',
      'default' => 'bg-primary-10 text-primary-70 before:bg-primary-50 hover:text-white disabled:hover:text-primary-70'
    ];

    $moodNavStyle = [
      'happy' => 'text-secondary-happy-30 hover:text-secondary-happy-70',
      'sad' => 'text-secondary-sad-30 hover:text-secondary-sad-70',
      'relaxed' => 'text-secondary-relaxed-30 hover:text-secondary-relaxed-70',
      'angry' => 'text-secondary-angry-30 hover:text-secondary-angry-70',
      'default' => 'text-primary-60 hover:text-primary-30'
    ];
@endphp
  
@if ($behaviour == 'action')
    <button 
      type='{{ $actionType }}' 
      {{ $attributes->merge(['class' => $baseStyle . ' ' . (($mood) ? $moodStyle[$mood] : $moodStyle['default']) . ' ']) }}
    >
      <span class='relative z-5'>{{ $content }}</span>
    </button>
@else
    <a href="{{ $navLink }}" class='inline'>
      @if ($navType == 'text')
        <p {{ $attributes->merge(['class' => 'w-fit font-bold text-smallBtn ' . (($mood) ? $moodNavStyle[$mood] : $moodNavStyle['default'])]) }}>
          {{ $content }}
        </p>
      @else
        <div {{ $attributes->merge(['class' => 'w-fit ' . $baseStyle . ' ' . (($mood) ? $moodStyle[$mood] : $moodStyle['default'])]) }}>
          <span class='relative z-5'>{{ $content }}</span>
        </div>
      @endif
    </a>
@endif