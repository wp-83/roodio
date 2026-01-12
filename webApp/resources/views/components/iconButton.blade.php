@props([
    'type',
    'arrowOrientation' => 'left'
])


@php
    // this part must be removed when page is ready
    $mood = 'relaxed';

    $iconTypes = [
        'cross', 'arrow', 'kebab', 'hamburger'
    ];

    $arrowRotate = [
        'left' => '',
        'top' => 'rotate-90',
        'right' => 'rotate-180',
        'bottom' => 'rotate-270'
    ];

    $elementCodeColor = [
        'happy' => '#FF8E2B',
        'sad' => '#6A4FBF',
        'relaxed' => '#28C76F',
        'angry' => '#E63946'
    ];

    $elementColor = [
        'happy' => 'bg-secondary-happy-70',
        'sad' => 'bg-secondary-sad-70',
        'relaxed' => 'bg-secondary-relaxed-70',
        'angry' => 'bg-secondary-angry-70'
    ];

    $backgroundHover = [
        'happy' => 'bg-secondary-happy-10 hover:bg-secondary-happy-30',
        'sad' => 'bg-secondary-sad-10 hover:bg-secondary-sad-30',
        'relaxed' => 'bg-secondary-relaxed-10 hover:bg-secondary-relaxed-30',
        'angry' => 'bg-secondary-angry-10 hover:bg-secondary-angry-30'
    ];
@endphp


<div class='w-max h-max cursor-pointer'>
    @if ($type === $iconTypes[0])
        <div {{ $attributes->merge(["class" => 'flex flex-col gap-5 w-8 h-8 group rounded-full relative border border-primary-50 '. $backgroundHover[$mood] . ' ']) }}>
            <div {{ $attributes->merge(["class" => 'w-5 h-0.5 absolute top-1/2 left-1/2 -translate-1/2 rotate-45 rounded-2xl '. $elementColor[$mood] . ' ']) }}></div>
            <div {{ $attributes->merge(["class" => 'w-5 h-0.5 absolute top-1/2 left-1/2 -translate-1/2 -rotate-45 rounded-2xl '. $elementColor[$mood] . ' ']) }}></div>
        </div>
    @elseif ($type === $iconTypes[1])
        <div {{ $attributes->merge(["class" => 'flex flex-col justify-center items-center gap-5 w-8 h-8 rounded-full relative ' . $backgroundHover[$mood] . ' ' . $elementColor[$mood] . ' ' . $arrowRotate[$arrowOrientation] . ' ']) }}>
            <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">        
            <!-- Uploaded to: SVG Repo, www.svgrepo.com, Transformed by: SVG Repo Mixer Tools -->
            <svg width="1.25rem" height="2.5rem" viewBox="0 -6.5 38 38" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" fill="{{ $elementCodeColor[$mood] }}">
                
                <g id="SVGRepo_bgCarrier" stroke-width="0"/>
                
                <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"/>

                    <g id="SVGRepo_iconCarrier"> <title>left-arrow</title> <desc>Created with Sketch.</desc> <g id="icons" stroke="none" stroke-width="1" fill="{{ $elementCodeColor[$mood] }}" fill-rule="evenodd"> <g id="ui-gambling-website-lined-icnos-casinoshunter" transform="translate(-1641.000000, -158.000000)" fill="{{ $elementCodeColor[$mood] }}" fill-rule="nonzero"> <g id="1" transform="translate(1350.000000, 120.000000)"> <path d="M317.812138,38.5802109 L328.325224,49.0042713 L328.41312,49.0858421 C328.764883,49.4346574 328.96954,49.8946897 329,50.4382227 L328.998248,50.6209428 C328.97273,51.0514917 328.80819,51.4628128 328.48394,51.8313977 L328.36126,51.9580208 L317.812138,62.4197891 C317.031988,63.1934036 315.770571,63.1934036 314.990421,62.4197891 C314.205605,61.6415481 314.205605,60.3762573 314.990358,59.5980789 L322.274264,52.3739093 L292.99947,52.3746291 C291.897068,52.3746291 291,51.4850764 291,50.3835318 C291,49.2819872 291.897068,48.3924345 292.999445,48.3924345 L322.039203,48.3917152 L314.990421,41.4019837 C314.205605,40.6237427 314.205605,39.3584519 314.990421,38.5802109 C315.770571,37.8065964 317.031988,37.8065964 317.812138,38.5802109 Z" id="left-arrow" transform="translate(310.000000, 50.500000) scale(-1, 1) translate(-310.000000, -50.500000) "> </path> </g> </g> </g> </g>

            </svg>
        </div>
    @elseif ($type === $iconTypes[2])
        <div {{ $attributes->merge(["class" => 'flex flex-col items-center justify-center gap-1 w-8 h-8 rounded-full '. $backgroundHover[$mood] . ' ']) }}>
            @for($i = 0; $i < 3; $i++)
                <div {{ $attributes->merge(["class" => 'w-1 h-1 rounded-full ' . $elementColor[$mood] . ' ']) }}></div>
            @endfor
        </div>
    @elseif ($type === $iconTypes[3])
        <div class='flex flex-col items-center justify-center gap-2 py-1 w-max h-max relative cursor-pointer z-10' id='hamburgerIcon'>
            @for($i = 0; $i < 3; $i++)
                <div {{ $attributes->merge(["class" =>  $elementColor[$mood] . ' w-8 h-1 rounded-md hamburger-line ']) }}></div>
            @endfor
            <div {{ $attributes->merge(["class" => 'absolute w-7 h-7 ']) }}>
                <img src="{{ asset('assets/icons/music-notes.svg') }}" alt="music-notes">
            </div>
        </div>
    @endif
</div>