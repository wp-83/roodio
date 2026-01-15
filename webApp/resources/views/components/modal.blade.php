@props([
    'modalId' => null,
    'isNeedBg' => true,
    'additionalStyle' => null
])

<div class='absolute top-0 left-0 z-100 hidden' id='{{ $modalId }}'>
    <div class='relative w-screen h-screen top-0 left-0 {{ ($isNeedBg) ? 'bg-shadedOfGray-100/15' : '' }}'>
        <div {{ 
            $attributes->merge([
                'class' => 'bg-white absolute min-w-5 h-max px-3 py-2 rounded-lg text-wrap ' . (($additionalStyle) ?? 'w-max') . ' '])    
        }}>
            <div class='w-full font-primary text-paragraph'>
                {{($header) ?? ''}}
            </div>
            <div class='w-full font-secondaryAndButton text-body-size'>
                {{($body) ?? ''}}
            </div>
            <div class='w-full mt-5 font-secondaryAndButton text-body-size'>
                {{($footer) ?? ''}}
            </div>
        </div>
    </div>
</div>

{{-- template for using, for isNeedBg must :isNeedBg
<x-modal modalId='' additionalStyle='' isNeedBg='true'>
    <x-slot name='header'></x-slot>
    <x-slot name='body'></x-slot>
    <x-slot name='footer'></x-slot>
</x-modal> --}}