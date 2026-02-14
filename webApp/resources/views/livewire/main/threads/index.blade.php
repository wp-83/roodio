<div>
    @push('head')
        <meta name="csrf-token" content="{{ csrf_token() }}">
    @endpush

    @push('script')
        <script src="{{ asset('js/pages/main/thread.js?v=2') }}" defer></script>
    @endpush


    @php
        $bgPlusIcon = [
            'happy' => 'bg-secondary-happy-10',
            'sad' => 'bg-secondary-sad-10',
            'relaxed' => 'bg-secondary-relaxed-10',
            'angry' => 'bg-secondary-angry-10'
        ];

        $borderPlusIcon = [
            'happy' => 'border-secondary-happy-85',
            'sad' => 'border-secondary-sad-85',
            'relaxed' => 'border-secondary-relaxed-85',
            'angry' => 'border-secondary-angry-85'
        ];

        $moodBaseColor = [
            'happy' => 'bg-secondary-happy-85',
            'sad' => 'bg-secondary-sad-85',
            'relaxed' => 'bg-secondary-relaxed-85',
            'angry' => 'bg-secondary-angry-85'
        ];

        $textColor = [
            'happy' => 'text-secondary-happy-30',
            'sad' => 'text-secondary-sad-30',
            'relaxed' => 'text-secondary-relaxed-30',
            'angry' => 'text-secondary-angry-30'
        ];

        $textMood = [
            'happy' => 'text-secondary-happy-100',
            'sad' => 'text-secondary-sad-100',
            'relaxed' => 'text-secondary-relaxed-100',
            'angry' => 'text-secondary-angry-100'
        ];

        $hoverStyle = [
            'happy' => 'not-placeholder-shown:bg-secondary-happy-10 border-secondary-happy-100',
            'sad' => 'not-placeholder-shown:bg-secondary-sad-10 border-secondary-sad-100',
            'relaxed' => 'not-placeholder-shown:bg-secondary-relaxed-10 border-secondary-relaxed-100',
            'angry' => 'not-placeholder-shown:bg-secondary-angry-10 border-secondary-angry-100'
        ];

        $checkboxStyle = [
            'happy' => 'peer-checked:bg-secondary-happy-100',
            'sad' => 'peer-checked:bg-secondary-sad-100',
            'relaxed' => 'peer-checked:bg-secondary-relaxed-100',
            'angry' => 'peer-checked:bg-secondary-angry-100'
        ];

        $popupBorder = [
            'happy' => 'border-secondary-happy-85',
            'sad' => 'border-secondary-sad-85',
            'relaxed' => 'border-secondary-relaxed-85',
            'angry' => 'border-secondary-angry-85'
        ];

        $errorStyle = 'bg-error-lighten/25 border-error-dark';
    @endphp

    
    <x-modal modalId='createThreadPopup' additionalStyle='relative z-1000 top-1/2 left-1/2 -translate-1/2 w-xs overflow-hidden border-3 {{ $popupBorder[$mood] }} md:w-md'>
        <x-slot name='header'>
            <div id='closeCreateThread' class='mb-3 w-max px-3 py-1 flex flex-row gap-2 {{ $bgPlusIcon[$mood] }} rounded-full cursor-pointer' wire:click="$dispatch('close-modal', {id: 'createThreadPopup'})"> {{-- Added wire:click for consistency if needed, though simple JS works too. Actually, let's stick to existing JS if possible, but form submit is now Livewire --}}
                <x-iconButton :mood='$mood' type='arrow'></x-iconButton>
                <p class='font-secondaryAndButton text-body-size font-bold text-primary-50'>Back</p>
            </div>
            <p class='font-bold text-paragraph md:text-subtitle text-primary-50 mt-3 mb-2 text-center'>Create Thread</p>
        </x-slot>
        <x-slot name='body'>
            <img src="{{ asset('assets/moods/icons/' . $mood . '.png') }}" alt="$mood" class='w-44 h-44 opacity-7 absolute left-0 bottom-0 translate-x-12 rotate-25 translate-y-4 group-hover:opacity-20 md:w-72 md:h-72 md:-translate-x-20 md:translate-y-18'>
            <form wire:submit="saveThread" id='createThread'>
                
                <div class="mb-6 flex flex-col font-secondaryAndButton">
                    <label for="title" class='mb-1 font-bold {{ $textMood[$mood] }}'>Title</label>
                    <input type="text" id='title' wire:model="title" class="border h-max p-1 px-2 text-small rounded-sm font-secondaryAndButton {{ $hoverStyle[$mood] . ' ' . (($errors->has('title')) ? $errorStyle : 'bg-shadedOfGray-10/40') }} " placeholder="Write a title...">
                    @error('title')
                        <p class='error-message'>{{ $message }}</p>
                    @enderror
                </div>
                <div class="mb-6 flex flex-col font-secondaryAndButton relative z-10">
                    <label for="content" class='mb-1 font-bold {{ $textMood[$mood] }}'>Content</label>
                    <textarea id='content' wire:model="content" rows='3' class="border rounded-sm p-2 resize-none scrollbar-none text-small {{ $hoverStyle[$mood] . ' ' . (($errors->has('content')) ? $errorStyle : 'bg-shadedOfGray-10/40') }}" placeholder="Share your thoughts..."></textarea>
                    @error('content')
                    <p class='error-message'>{{ $message }}</p>
                    @enderror
                </div>
                <div class="mb-5">
                    <label class="inline-flex justify-center items-center cursor-pointer">
                        <input type="checkbox" wire:model="isReplyable" class="sr-only peer">
                        <div class="relative w-9 h-5 bg-shadedOfGray-30 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-brand-soft rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-buffer after:content-[''] after:absolute after:top-0.5 after:start-0.5 after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all {{ $checkboxStyle[$mood] }}"></div>
                        <span class="select-none ms-2 text-small font-secondaryAndButton text-shadedOfGray-100">Can be reply by others</span>
                    </label>
                </div>
                <x-button :mood='$mood' actionType='submit' id='createThreadBtn' content='Send your reply'></x-button>
            </form>
        </x-slot>
    </x-modal>

    <div class='mb-5 md:mb-9'>
        <div class='flex w-full flex-row justify-between items-center'>
            <div class='flex flex-row gap-3 items-center justify-center'>
                <img src="{{ asset('assets/moods/' . $mood . '.png') }}" alt="{{ $mood }}" class='h-26 w-26 md:h-32 md:w-32 lg:h-40 lg:w-40'>
                <div class='flex flex-col'>
                    <div class='w-0 relative overflow-hidden typingTextAnimation max-w-max '>
                        <p class='font-primary text-title font-bold {{ $textColor[$mood] }} md:text-hero' >Threads</p>
                    </div>
                    <p class='font-secondaryAndButton text-white text-justify contentFadeLoad text-small md:text-body-size'>Nice to meet you again, {{ Str::before($fullname, ' ') }}! Go ahead and express how you feel.</p>
                </div>
            </div>
            <div class='w-max hidden lg:inline contentFadeLoad'>
                <a class='createThreadBtn cursor-pointer'>
                    <x-button content='Add new thread' :mood='$mood'></x-button>
                </a>
            </div>
        </div>
    </div>
    <a class='group createThreadBtn cursor-pointer'>
        <div class='absolute z-75 right-7 bottom-7' id='threadIcon'>
            <div class="relative w-12 h-12 rounded-full border-2 bg-white {{ $borderPlusIcon[$mood] }}">
                <div class='absolute w-8 h-1 {{ $moodBaseColor[$mood] }} rounded-full top-1/2 left-1/2 -translate-1/2 md:h-1.25'></div>
                <div class='absolute w-8 h-1 {{ $moodBaseColor[$mood] }} rounded-full top-1/2 left-1/2 -translate-1/2 rotate-90 md:h-1.25'></div>
            </div>
        </div>
        <div id='threadIconLabel' class='z-10 absolute opacity-0 transition-opacity duration-150 h-max right-12 bottom-9.5 pl-2 pr-10 text-primary-70 rounded-md py-0.5 text-small border-2 {{ $bgPlusIcon[$mood] . ' ' . $borderPlusIcon[$mood] }} group-hover:opacity-100 '>
            <p>Give your opinion!</p>
        </div>
    </a>
    
    {{-- Search and Filters --}}
    {{-- <form action="{{ route('threads.index') }}" method="GET">  REMOVE FORM --}}
        
        <div class='mb-7 flex flex-row gap-3 w-full lg:justify-end contentFadeLoad'>
             {{-- Using standard inputs or updating filter buttons to use Livewire --}}
             {{-- Since x-filterButton uses standard input radio logic, we can stick to that but bind it to wire:model --}}
             
             {{-- We might need to adjust x-filterButton to support wire:click or wire:model --}}
             {{-- let's try direct wire:click on the component call if it accepts attributes, or wrapping it --}}
             
             <div wire:click="$set('filter', 'all')" class="cursor-pointer">
                <x-filterButton id='allFilter' name='filter' value='all' :mood='$mood' label='All' :active="$filter === 'all'"></x-filterButton>
             </div>
             <div wire:click="$set('filter', 'following')" class="cursor-pointer">
                 <x-filterButton id='followingFilter' name='filter' value='following' :mood='$mood' label='Following' :active="$filter === 'following'"></x-filterButton>
             </div>
             <div wire:click="$set('filter', 'created')" class="cursor-pointer">
                 <x-filterButton id='createdFilter' name='filter' value='created' :mood='$mood' label='My Threads' :active="$filter === 'created'"></x-filterButton>
             </div>

             {{-- Hidden search input to bind to query string if needed --}}
             @if($search)
                 <input type="hidden" wire:model="search" value="{{ $search }}">
             @endif
        </div>
    {{-- </form> --}}

    <div class='columns-1 md:columns-2 lg:columns-3 2xl:columns-4 gap-5 2xl:gap-3 contentFadeLoad'>
        @forelse($threads as $thread)
            <div class='break-inside-avoid mb-5'>
                <livewire:user.thread-box
                    :thread="$thread"
                    :mainUser="$user"
                    :mood="$mood"
                    :isReplyable="$thread->isReplyable"
                    :wire:key="'thread-box-'.$thread->id"
                />
            </div>
        @empty
            <p class='text-white font-secondaryAndButton text-small md:text-body-size'>There is no thread posted.</p>
        @endforelse
    </div>
</div>
