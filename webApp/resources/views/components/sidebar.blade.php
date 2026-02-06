@props([
    'mood'
])

{{-- belom passing route untuk masing2 button --}}
<div
{{
    $attributes->merge([
        'class' => 'flex flex-col gap-5 w-fit bg-primary-70 h-full pt-2 -translate-x-full md:translate-x-0 transition-transform md:transition-[width] duration-350 '
    ])
}} id='sidebar'
>
    <x-sidebarButton route="user.index" :mood="$mood" icon='home' label='Home' content="Let's play the music!"></x-sidebarButton>
    <x-sidebarButton route='thread.index' :params="['filter' => 'all']" :mood="$mood" icon='forum' label='Threads' content="Be part of the discussion!"></x-sidebarButton>
    <x-sidebarButton route="social.index" :params="['filter' => 'all']" :mood="$mood" icon='social' label='Socials' content="Connect with others now!"></x-sidebarButton>
    <x-sidebarButton :mood="$mood" icon='mood' label='Moods' content="Let's see your mood history!"></x-sidebarButton>
</div>
