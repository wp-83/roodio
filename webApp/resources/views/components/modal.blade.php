@props([
    'modalId' => null,
    'isNeedBg' => true,
    'additionalStyle' => null,
    'centered' => false
])

<div class='fixed inset-0 z-[100] invisible opacity-0 transition-all duration-300 {{ $centered ? "flex items-center justify-center" : "" }}' id='{{ $modalId }}'>
    <div class='absolute inset-0 {{ ($isNeedBg) ? 'bg-shadedOfGray-100/35' : '' }}'></div>
    <div {{ 
        $attributes->merge([
            'class' => 'popupContent min-w-5 h-max p-5 rounded-lg text-wrap bg-white z-10 ' . 
                       ($centered ? 'relative' : 'absolute') . ' ' . 
                       (($additionalStyle) ?? ' w-max') . ' '])
    }}>
        <div class='w-full font-primary text-body-size lg:text-paragraph'>
            {{($header) ?? ''}}
        </div>
        <div class='w-full font-secondaryAndButton text-small lg:text-body-size'>
            {{($body) ?? ''}}
        </div>
        <div class='w-full font-secondaryAndButton text-small lg:text-body-size'>
            {{($footer) ?? ''}}
        </div>
    </div>
</div>

{{-- template for using
<x-modal modalId='' additionalStyle='' :isNeedBg='true'>
    <x-slot name='header'></x-slot>
    <x-slot name='body'></x-slot>
    <x-slot name='footer'></x-slot>
</x-modal> --}}