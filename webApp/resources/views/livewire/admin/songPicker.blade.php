<div>
    <label class="block font-secondaryAndButton text-small text-primary-100 font-bold mb-3">
        Pilih Lagu
    </label>

    <div class="relative mb-3">
        <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-shadedOfGray-60">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
        </span>
        <input wire:model.live.debounce.300ms="search"
               type="text"
               placeholder="Cari lagu atau artis..."
               class="w-full pl-10 pr-4 py-2 bg-white border border-shadedOfGray-30 rounded-lg text-primary-100 font-secondaryAndButton text-body-size focus:border-accent-100 focus:ring-1 focus:ring-accent-100 outline-none transition placeholder-shadedOfGray-30">
    </div>

    <div class="border border-shadedOfGray-20 rounded-lg overflow-hidden max-h-60 overflow-y-auto custom-scrollbar bg-shadedOfGray-10/10 relative">

        <div wire:loading wire:target="search" class="absolute inset-0 bg-white/80 z-10 flex items-center justify-center">
            <span class="text-primary-60 text-small font-bold animate-pulse">Mencari...</span>
        </div>

        <ul class="divide-y divide-shadedOfGray-10">
            @forelse($songs as $song)
                <li class="flex items-center justify-between p-3 hover:bg-shadedOfGray-10/50 transition-colors cursor-pointer group" wire:key="song-{{ $song->id }}">
                    <label class="flex items-center space-x-4 w-full cursor-pointer">
                        <div class="flex-shrink-0">
                            <input type="checkbox"
                                   wire:model.live="selectedSongs"
                                   value="{{ $song->id }}"
                                   class="form-checkbox h-5 w-5 text-primary-60 border-shadedOfGray-30 rounded focus:ring-primary-50 transition duration-150 ease-in-out">
                        </div>
                        <div class="flex-grow">
                            <p class="font-primary font-bold text-primary-100 text-body-size group-hover:text-primary-60 transition-colors">
                                {{ $song->title }}
                            </p>
                            <p class="font-secondaryAndButton text-small text-shadedOfGray-60">
                                {{ $song->artist }}
                            </p>
                        </div>
                    </label>
                </li>
            @empty
                <li class="p-4 text-center text-shadedOfGray-50 text-small italic">
                    Tidak ada lagu ditemukan.
                </li>
            @endforelse
        </ul>
    </div>

    <div class="flex justify-between items-center mt-2">
        <p class="text-micro text-shadedOfGray-40 italic">*Scroll untuk melihat lebih banyak.</p>
        <p class="text-small font-bold text-primary-60">{{ count($selectedSongs) }} Lagu dipilih</p>
    </div>

    @foreach($selectedSongs as $songId)
        <input type="hidden" name="songs[]" value="{{ $songId }}">
    @endforeach

</div>
