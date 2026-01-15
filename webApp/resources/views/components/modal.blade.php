@props([
    'modalId' => null,
    'additionalStyle' => null
])

<div class='absolute top-0 left-0 z-100 hidden' id='{{ $modalId }}'>
    <div class='relative w-screen h-screen bg-shadedOfGray-100/15 top-0 left-0'>
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