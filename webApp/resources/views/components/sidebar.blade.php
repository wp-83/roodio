@vite(['resources/css/app.css', 'resources/js/app.js'])

@props([
    'mood' => 'relaxed',
    'isToggle' => true
])

{{-- belom passing route untuk masing2 button --}}
<div
{{ 
    $attributes->merge([
        'class' => 'flex flex-col gap-5 w-fit bg-primary-85 h-full pt-10 ' . (($isToggle) ? 'px-4 ' : ' ')
    ])
}}
>
    <x-sidebarButton mood="{{ $mood }}" icon='home' label='Home' content="Let's play the music!" route='awikwok' isToggle='{{ $isToggle }}'></x-sidebarButton>
    <x-sidebarButton mood="{{ $mood }}" icon='forum' label='Forum' content="Be part of the discussion!" isToggle='{{ $isToggle }}'></x-sidebarButton>
    <x-sidebarButton mood="{{ $mood }}" icon='social' label='Social' content="Connect with others now!" isToggle='{{ $isToggle }}'></x-sidebarButton>
    <x-sidebarButton mood="{{ $mood }}" icon='mood' label='Mood' content="Let's see your mood history!" isToggle='{{ $isToggle }}'></x-sidebarButton>
</div>