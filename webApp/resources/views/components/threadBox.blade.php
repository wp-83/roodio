@props([
    'mood',
    'creator',
    'createdAt',
    'title',
    'content',
    'thread',
    'profilePicture',
    'isReplyable',
    'mainUser'
])

@php
    $isFollowing = false;

    $bgContainer = [
        'happy' => 'bg-secondary-happy-10/95',
        'sad' => 'bg-secondary-sad-10/95',
        'relaxed' => 'bg-secondary-relaxed-10/95',
        'angry' => 'bg-secondary-angry-10/95',
    ];

    $elementColor = [
        'happy' => '#FF8E2B',
        'sad' => '#6A4FBF',
        'relaxed' => '#28C76F',
        'angry' => '#E63946'
    ];

    $borderMood = [
        'happy' => 'border-secondary-happy-100',
        'sad' => 'border-secondary-sad-100',
        'relaxed' => 'border-secondary-relaxed-100',
        'angry' => 'border-secondary-angry-100'
    ];

    $textMood = [
        'happy' => 'text-secondary-happy-100',
        'sad' => 'text-secondary-sad-100',
        'relaxed' => 'text-secondary-relaxed-100',
        'angry' => 'text-secondary-angry-100',
    ];
@endphp

<div class='{{ $bgContainer[$mood] }} rounded-lg h-max p-5 w-full'>
    <div class='flex flex-row items-center gap-2 w-full'>
        <div class='w-16 h-16 lg:w-18 lg:h-18 border-2 {{ $borderMood[$mood] }} rounded-full flex items-center justify-center'>
            <div class='w-14 h-14 lg:w-16 lg:h-16 bg-primary-10 rounded-full flex items-center justify-center relative z-5 overflow-hidden'>
            @if (!empty($profilePicture))
                <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePicture }}" alt="{{ $creator }}" class='w-full h-full object-cover'>
            @else
                <p class='text-subtitle font-primary font-bold text-primary-70 h-fit'>{{ Str::charAt(Str::upper($creator), 0) }}</p>
            @endif
        </div>
        </div>
        <div class='flex flex-row justify-between items-center flex-1 w-full font-secondaryAndButton'>
            <div class='flex flex-col'>
                <p class='text-body-size font-bold {{ $textMood[$mood] }}'>{{ ($mainUser->id == $thread->userId) ? 'You' : $creator }}</p>
                <p class='text-micro lg:text-small'>{{ $createdAt }}</p>
            </div>
            @if (!($mainUser->id == $thread->userId))
                <livewire:user.button-follow :thread="$thread"/>
            @endif
        </div>
    </div>
    <hr class='border rounded-full my-3 {{ $borderMood[$mood] }}'>
    <div class='mb-6'>
        <p class='font-bold font-primary text-primary-60 text-paragraph lg:text-subtitle'>{{ $title }}</p>
        <p class='font-secondaryAndButton text-small lg:text-body-size'>{{ $content }}</p>
        {{-- Str::limit($content, 1000, '...') --}}
    </div>
    <div class='flex flex-row gap-8 items-center'>
        <livewire:user.reaction-button :thread-id="$thread->id" />
        @if ($isReplyable)
            <div class='flex gap-1 items-center'>
                <div class='w-6 h-6'>
                    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

                        <!-- Uploaded to: SVG Repo, www.svgrepo.com, Transformed by: SVG Repo Mixer Tools -->
                        <svg width="100%" height="100%" viewBox="0 0 32 32" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:sketch="http://www.bohemiancoding.com/sketch/ns" fill="{{ $elementColor[$mood] }}">

                        <g id="SVGRepo_bgCarrier" stroke-width="0"/>

                        <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"/>

                        <g id="SVGRepo_iconCarrier"> <title>comment 3</title> <desc>Created with Sketch Beta.</desc> <defs> </defs> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd" sketch:type="MSPage"> <g id="Icon-Set-Filled" sketch:type="MSLayerGroup" transform="translate(-207.000000, -257.000000)" fill="{{ $elementColor[$mood] }}"> <path d="M231,273 C229.896,273 229,272.104 229,271 C229,269.896 229.896,269 231,269 C232.104,269 233,269.896 233,271 C233,272.104 232.104,273 231,273 L231,273 Z M223,273 C221.896,273 221,272.104 221,271 C221,269.896 221.896,269 223,269 C224.104,269 225,269.896 225,271 C225,272.104 224.104,273 223,273 L223,273 Z M215,273 C213.896,273 213,272.104 213,271 C213,269.896 213.896,269 215,269 C216.104,269 217,269.896 217,271 C217,272.104 216.104,273 215,273 L215,273 Z M223,257 C214.164,257 207,263.269 207,271 C207,275.419 209.345,279.354 213,281.919 L213,289 L220.009,284.747 C220.979,284.907 221.977,285 223,285 C231.836,285 239,278.732 239,271 C239,263.269 231.836,257 223,257 L223,257 Z" id="comment-3" sketch:type="MSShapeGroup"> </path> </g> </g> </g>

                        </svg>
                </div>
                <p class='font-secondaryAndButton text-primary-60 text-micro md:text-small'>10.3K</p>
            </div>
        @endif
    </div>
    @if ($isReplyable)
        <div class='mt-6'>
            <hr class='border border-shadedOfGray-30 my-2'>
            <div>
                <div id="replyContainer" class="bg-primary-20 h-36 overflow-y-auto">
                    @forelse($thread->replies as $reply)
                        <p>{{ $reply->content }}</p>
                    @empty
                    @endforelse
                </div>
                    <div class="">
                        <form action="{{ route('thread.reply', $thread->id) }}" method="POST">
                            @csrf
                            <label for="content">Reply:</label>
                            <textarea name="content" class="border"></textarea>
                            <button type="submit">send</button>
                        </form>
                        @error('content')
                            {{ $message }}
                        @enderror
                    </div>
            </div>
        </div>
    @endif
</div>
