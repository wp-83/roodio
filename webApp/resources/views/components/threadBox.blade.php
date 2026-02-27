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

    $bgMoodStyle = [
        'happy' => 'bg-secondary-happy-30',
        'sad' => 'bg-secondary-sad-30',
        'relaxed' => 'bg-secondary-relaxed-30',
        'angry' => 'bg-secondary-angry-30'
    ];

    $scrollbarTheme = [
        'happy' => 'scrollbar-thumb-secondary-happy-85/75 scrollbar-track-transparent',
        'sad' => 'scrollbar-thumb-secondary-sad-85/75 scrollbar-track-transparent',
        'relaxed' => 'scrollbar-thumb-secondary-relaxed-85/75 scrollbar-track-transparent',
        'angry' => 'scrollbar-thumb-secondary-angry-85/75 scrollbar-track-transparent'
    ];

    $borderTextareaStyle = [
        'happy' => 'border-secondary-happy-100',
        'sad' => 'border-secondary-sad-100',
        'angry' => 'border-secondary-angry-100',
        'relaxed' => 'border-secondary-relaxed-100'
    ];
@endphp


<div class='{{ $bgContainer[$mood] }} relative rounded-lg h-max p-5 w-full overflow-hidden lg:max-w-md'>
    <img src="{{ asset('assets/moods/icons/' . $mood . '.png') }}" alt="$mood" class='w-48 h-48 opacity-15 absolute right-0 top-0 translate-x-14 -rotate-145 -translate-y-13 md:w-68 md:h-68 md:translate-x-18 md:-translate-y-24'>
    <div class='flex flex-row items-center gap-2 w-full'>
        <div class='w-16 h-16 lg:w-18 lg:h-18 border-2 {{ $borderMood[$mood] }} rounded-full flex items-center justify-center'>
            <div class='w-14 h-14 lg:w-16 lg:h-16 bg-primary-10 rounded-full flex items-center justify-center relative z-5 overflow-hidden'>
            @if (!empty($profilePicture))
                <img src="{{ config('filesystems.storage_url') . '/' . $profilePicture }}" alt="{{ $creator }}" class='w-full h-full object-cover'>
            @else
                <p class='text-subtitle font-primary font-bold text-primary-70 h-fit'>{{ Str::charAt(Str::upper($creator), 0) }}</p>
            @endif
        </div>
        </div>
        <div class='flex flex-row justify-between items-center flex-1 w-full font-secondaryAndButton'>
            <div class='flex flex-col'>
                <p class='text-body-size font-bold {{ $textMood[$mood] }}'>{{ ($mainUser->id == $thread->userId) ? 'You' : Str::limit($creator, 18) }}</p>
                <p class='text-micro lg:text-small'>{{ $createdAt }}</p>
            </div>
            @if (!($mainUser->id == $thread->userId))
                <livewire:user.button-follow
                    :userId="$thread->userId"
                    :mood="$mood"
                    :wire:key="'follow-thread-'.$thread->id"
                    customStyle="zoom:0.75"
                />
            @endif
        </div>
    </div>
    <hr class='border rounded-full my-3 {{ $borderMood[$mood] }}'>
    <div class='mb-6 w-full'>
        <div class='w-full overflow-hidden'>
            <p class='font-bold font-primary text-primary-60 text-body-size lg:text-paragraph'>{{ $title }}</p>
            <p class='font-secondaryAndButton text-small lg:text-body-size'>{!! nl2br(e($content)) !!}</p>
        </div>
    </div>
    <div class='w-full relative'>
        <div class='flex flex-row gap-4'>
            <livewire:user.reaction-button :thread-id="$thread->id" />
            @if ($isReplyable)
                <div class='flex gap-1 items-center' id='toggleComment' data-thread="reply-{{ $thread->id }}">
                    <div class='w-6 h-6 cursor-pointer'>
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
            <div class='mt-6 w-full hidden' id="reply-{{ $thread->id }}">
                <hr class='border border-shadedOfGray-30 my-2'>
                <div class="h-max max-h-64 mt-3 overflow-y-auto replyContainer font-secondaryAndButton scrollbar-thin {{ $scrollbarTheme[$mood] }}">
                    @forelse($thread->replies as $reply)
                        @php
                            $profilePhoto = $reply->user->userDetail->profilePhoto;
                            $fullname = $reply->user->userDetail->fullname;
                        @endphp
                        <div class='mb-3 flex flex-row items-start gap-3'>
                            <div class='w-8 h-8 rounded-full flex items-center justify-center relative z-5 overflow-hidden {{ $bgMoodStyle[$mood] }}'>
                                @if (!empty($profilePhoto))
                                    <img src="{{ config('filesystems.storage_url') . '/' . $profilePhoto }}" alt="{{ $fullname }}" class='w-full h-full object-cover'>
                                @else
                                    <p class='text-small font-primary font-bold h-fit {{ $textMood[$mood] }}'>{{ Str::charAt(Str::upper($fullname), 0) }}</p>
                                @endif
                            </div>
                            <div class='flex flex-col gap-1.25'>
                                <div>
                                    <p class='text-small font-bold {{ $textMood[$mood] }}'>{{ ($mainUser->id == $reply->userId) ? 'You' : $reply->user->userDetail->fullname }}</p>
                                    <p class='text-micro text-shadedOfGray-50 italic'>{{ \Carbon\Carbon::parse($reply->created_at)->diffForHumans() }}</p>
                                </div>
                                <p class='text-small'>{!! nl2br(e($reply->content)) !!}</p>
                            </div>
                        </div>
                    @empty
                    @endforelse
                </div>
                <div class="w-full mt-4 relative">
                    <form action="{{ route('threads.reply', $thread->id) }}" method="POST">
                        @csrf
                        <textarea name='content' rows='1' placeholder="Reply this thread..." class="font-secondaryAndButton text-small w-full min-h-1 max-h-18 p-2 pl-6 pr-20 py-3 overflow-y-auto resize-none border {{ $borderTextareaStyle[$mood] }} bg-shadedOfGray-10/60 not-placeholder-shown:bg-accent-85/10 rounded-3xl scrollbar-none placeholder:italic" oninput="this.style.height='auto'; this.style.height=this.scrollHeight+'px';"></textarea>
                        <div class='absolute top-1 right-2'>
                            <x-button actionType='submit' :mood='$mood' content='Send' class='w-max absolute top-0 right-0' style='zoom:0.7;'></x-button>
                        </div>
                    </form>
                    @error('content')
                        <p class='font-secondaryAndButton error-message pt-0.1 mb-2'>{{ $message }}</p>
                    @enderror
                </div>
            </div>
        @endif
    </div>
</div>
