@props([
    'type' => 'text',
    'id',
    'name' => null,
    'icon' => null,
    'label' => null,
    'additionalInfo' => null,
    'isRequired' => true,
    'placeholder' => null
])

{{-- type checkbox dan radio --}}

@php
    $name = $name ?? $id;

    $baseStyle = 'w-full px-1.5 py-0.5 h-8 text-small border-b-2 rounded-md not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/65 hover:bg-shadedOfGray-30/45 ease-in-out duration-125 md:text-body-size md:placeholder:text-small md:h-9';

    $normalStyle = 'bg-shadedOfGray-20/50 border-shadedOfGray-50';
    $errorStyle = 'bg-error-lighten/25 border-error-dark';
    $conditionalStyle = ($errors->has($name)) ? $errorStyle : $normalStyle;
    
    $typeStyle = '';
    switch ($type) {
        case 'password':
            $typeStyle = 'pr-8';
            break;

        default:
            $typeStyle = '';
            break;
    }
@endphp

@if ($type == 'search')
    <div class='flex flex-row items-center px-2 py-0.5 h-10 text-small rounded-full placeholder:text-micro placeholder:italic bg-shadedOfGray-10 hover:bg-white focus-within:bg-white ease-in-out duration-125 md:text-body-size md:placeholder:text-small md:h-9 group'>
        <img src="{{ asset('assets/icons/search.svg') }}" alt="search" class='w-8 h-8 pr-1 mr-2 border-r-2 border-primary-70'>
        <input type="text" autocomplete="off" id='search' placeholder='Search songs, artist, lyrics' class='w-full pr-2'>
        <x-iconButton type='cross' mood='relaxed' style='zoom:0.8;' class='invisible group-focus-within:visible' id='searchClose'></x-iconButton>
    </div>
@else
    <div class='flex flex-col p-1 h-max mb-6 w-full'>
        <label for="{{ $name }}" class='relative w-full text-small mb-1 md:text-body-size'>
            <div class='flex flex-row justify-between'>
                <div class='flex flex-row'>
                    @isset($icon)
                        <img src="{{ asset('assets/icons/'. $icon .'.svg') }}" alt='{{ $icon }}' class='w-5 mr-1.5 md:w-6 lg:w-7'>
                    @endisset

                    @isset($label)
                        <p class='text-primary-85'>
                            {{ $label }}@if ($isRequired)<span class='text-error-dark'>*</span>@endif
                        </p>
                    @endisset
                </div>

                @isset($inlineContent)
                    {{ $inlineContent }}
                @endisset
            </div>

            @isset($additionalLabelButton)
                {{ $additionalLabelButton }}
            @endisset
        </label>

        @isset($additionalInfo)
            <p class='text-micro text-shadedOfGray-70 font-bold md:text-small'>{{ $additionalInfo }}</p>
        @endisset

        <div class='relative'>
            <input
                type="{{ $type }}"
                id="{{ $id }}"
                name="{{ $name }}"
                autocomplete="{{ ($attributes->has('autocomplete')) ? 'on' : 'off' }}"
                @isset($placeholder)
                    placeholder="{{ $placeholder }}"
                @endisset
                {{ $attributes->merge([
                    'class' => $baseStyle . ' ' . $typeStyle . ' ' . $conditionalStyle
                ]) }}
            />
            @isset($additionalContent)
                {{ $additionalContent }}
            @endisset
        </div>

        <div class="error-message">
            @error($name)
                {{ $message }}
            @enderror
        </div>
    </div>
@endif