@vite(['resources/css/app.css', 'resources/js/app.js'])

{{-- this part should be removed when page is ready --}}
@php
    $mood = 'relaxed';
@endphp

{{-- belom passing route untuk masing2 button --}}
<div
{{ 
    $attributes->merge([
        'class' => 'flex flex-col gap-5 w-fit bg-primary-70 h-full pt-2 -translate-x-full md:translate-x-0 '
    ])
}} id='sidebar'
>
    <x-sidebarButton mood="{{ $mood }}" icon='home' label='Home' content="Let's play the music!"></x-sidebarButton>
    <x-sidebarButton mood="{{ $mood }}" icon='forum' label='Forum' content="Be part of the discussion!"></x-sidebarButton>
    <x-sidebarButton mood="{{ $mood }}" icon='social' label='Social' content="Connect with others now!"></x-sidebarButton>
    <x-sidebarButton mood="{{ $mood }}" icon='mood' label='Mood' content="Let's see your mood history!"></x-sidebarButton>
</div>