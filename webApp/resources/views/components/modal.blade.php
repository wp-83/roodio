@props([
    'modalId' => null,
    'isNeedBg' => true,
    'additionalStyle' => null
])

<div class='absolute top-0 left-0 z-100 ' id='{{ $modalId }}'>
    <div class='relative w-screen h-screen top-0 left-0  {{ ($isNeedBg) ? 'bg-shadedOfGray-100/30' : '' }}'>
        <div {{ 
            $attributes->merge([
                'class' => 'absolute min-w-5 h-max p-5 rounded-lg text-wrap bg-white ' . (($additionalStyle) ?? ' w-max') . ' '])
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
</div>

{{-- template for using
<x-modal modalId='' additionalStyle='' :isNeedBg='true'>
    <x-slot name='header'></x-slot>
    <x-slot name='body'></x-slot>
    <x-slot name='footer'></x-slot>
</x-modal> --}}