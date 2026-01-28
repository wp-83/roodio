<div class="space-y-4">

    {{-- Search Bar (Sticky Top agar tidak hilang saat scroll) --}}
    <div class="relative sticky top-0 z-20 bg-primary-100 pb-2">
        <span class="absolute inset-y-0 left-0 flex items-center pl-4 text-shadedOfGray-40 pb-2">
            <i class="fa-solid fa-magnifying-glass"></i>
        </span>
        <input wire:model.live.debounce.300ms="search"
               type="text"
               placeholder="Search song title or artist..."
               class="w-full bg-primary-85 border border-primary-60 text-white text-sm rounded-xl pl-11 pr-4 py-3 focus:outline-none focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 placeholder-shadedOfGray-50 transition-all shadow-inner"
               autofocus>

        {{-- Loading Indicator (Kecil di pojok kanan input) --}}
        <div wire:loading wire:target="search" class="absolute right-4 top-3 text-secondary-happy-100 text-xs font-bold animate-pulse">
            Searching...
        </div>
    </div>

    {{-- Song List Grid --}}
    <div class="grid grid-cols-1 gap-2">
        @forelse($songs as $song)
            <div wire:key="song-{{ $song->id }}"
                 class="group flex items-center justify-between p-3 bg-primary-85 hover:bg-primary-70 border border-transparent hover:border-primary-60 rounded-xl transition-all duration-200">

                {{-- Song Info --}}
                <div class="flex items-center gap-4 min-w-0">
                    {{-- Icon Music --}}
                    <div class="w-10 h-10 rounded-lg bg-primary-60 flex items-center justify-center text-shadedOfGray-30 group-hover:text-white transition-colors flex-shrink-0">
                        <i class="fa-solid fa-music"></i>
                    </div>

                    {{-- Text --}}
                    <div class="min-w-0">
                        <h4 class="text-sm font-bold text-white truncate pr-2 group-hover:text-secondary-happy-100 transition-colors">
                            {{ $song->title }}
                        </h4>
                        <p class="text-xs text-shadedOfGray-40 truncate">
                            {{ $song->artist }}
                        </p>
                    </div>
                </div>

                {{-- Action Button --}}
                <button type="button"
                        wire:click.prevent="selectSong('{{ $song->id }}')"
                        wire:loading.attr="disabled"
                        wire:target="selectSong('{{ $song->id }}')"
                        class="px-4 py-2 bg-primary-60 hover:bg-secondary-happy-100 text-white text-xs font-bold rounded-lg transition-all shadow-md flex items-center gap-2 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">

                    {{-- State Normal --}}
                    <span wire:loading.remove wire:target="selectSong('{{ $song->id }}')">
                        Add
                    </span>
                    <i wire:loading.remove wire:target="selectSong('{{ $song->id }}')" class="fa-solid fa-plus"></i>

                    {{-- State Loading --}}
                    <span wire:loading wire:target="selectSong('{{ $song->id }}')">
                        Adding...
                    </span>
                    <i wire:loading wire:target="selectSong('{{ $song->id }}')" class="fa-solid fa-spinner fa-spin"></i>
                </button>
            </div>
        @empty
            {{-- Empty State --}}
            <div class="text-center py-12 flex flex-col items-center justify-center border-2 border-dashed border-primary-60 rounded-xl">
                <div class="w-12 h-12 rounded-full bg-primary-85 flex items-center justify-center mb-3 text-shadedOfGray-40">
                    <i class="fa-solid fa-music-slash text-xl"></i>
                </div>
                <p class="text-sm text-shadedOfGray-30 font-medium">No songs found.</p>
                <p class="text-xs text-shadedOfGray-50 mt-1">Try searching with a different keyword.</p>
            </div>
        @endforelse
    </div>
</div>
