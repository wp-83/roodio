@vite(['resources/css/app.css', 'resources/js/app.js'])

@props([
  'behaviour' => 'navigation', //action OR navigation
  'actionType' => 'button',
  'navLink' => null,
  'content' => 'hallo ini button',
  'mood' => null //happy, sad, relaxed, angry
])

@php
    $navLink ??= route('welcome');

    $baseStyle = 'relative overflow-hidden w-max px-4 py-1 font-secondaryAndButton rounded-3xl cursor-pointer text-md text-center font-bold before:absolute before:inset-0 before:-translate-x-full before:transition-transform before:duration-300 hover:before:translate-x-0 md:text-smallBtn md:px-5 md:py-1.5';

    $defaultStyle = 'bg-primary-10 text-primary-85 before:bg-primary-50 hover:text-white';
    $happyMood = 'bg-secondary-happy-20 text-secondary-happy-100 before:bg-secondary-happy-85 hover:text-secondary-happy-10';
    $sadMood = 'bg-secondary-sad-20 text-secondary-sad-100 before:bg-secondary-sad-85 hover:text-secondary-sad-10';
    $relaxedMood = 'bg-secondary-relaxed-20 text-secondary-relaxed-100 before:bg-secondary-relaxed-85 hover:text-secondary-relaxed-10';
    $angryMood = 'bg-secondary-angry-20 text-secondary-angry-100 before:bg-secondary-angry-85 hover:text-secondary-angry-10';

    switch ($mood) {
      case 'happy':
        $conditionalStyle = $happyMood;
        break;
      
      case 'sad':
        $conditionalStyle = $sadMood;
        break;
      
      case 'relaxed':
        $conditionalStyle = $relaxedMood;
        break;
      
      case 'angry':
        $conditionalStyle = $angryMood;
        break;
      
      default:
        $conditionalStyle = $defaultStyle;
        break;
    };
@endphp
  

@if ($behaviour == 'action')
    <button 
      type='{{ $actionType }}' 
      {{ $attributes->merge([
        'class' => $baseStyle . ' ' . $conditionalStyle
      ]) }}
    >
      <span class='relative z-5'>{{ $content }}</span>
    </button>
@else
    <a href="{{ $navLink }}">
      <div {{ $attributes->merge(['class' => $baseStyle . ' ' . $conditionalStyle]) }}>
        <span class='relative z-5'>{{ $content }}</span>
      </div>
    </a>
@endif