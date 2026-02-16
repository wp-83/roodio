<div class="space-y-3">

    {{-- Jika List Kosong --}}
    @if(empty($selectedSongs))
        <div class="text-center py-8 border-2 border-dashed border-primary-70 rounded-xl bg-primary-85/50">
            <i class="fa-solid fa-music text-3xl text-[#9CA3AF] mb-2"></i>
            <p class="text-sm text-shadedOfGray-30">No tracks selected yet.</p>
            <p class="text-xs text-shadedOfGray-50">Click "Add Tracks" to browse library.</p>
        </div>
    @else
        {{-- Loop Lagu Terpilih --}}
        @foreach($selectedSongs as $index => $song)
            <div class="flex items-center justify-between p-3 bg-primary-100 border border-primary-60 rounded-xl group hover:border-secondary-happy-100 transition-colors">

                <div class="flex items-center gap-3 overflow-hidden">
                    {{-- Icon / Cover Kecil --}}
                    <div class="w-10 h-10 rounded-lg bg-primary-70 flex items-center justify-center text-shadedOfGray-30 flex-shrink-0">
                        <i class="fa-solid fa-music"></i>
                    </div>

                    {{-- Info Lagu --}}
                    <div class="min-w-0">
                        <h4 class="text-sm font-bold text-white truncate">{{ $song->title }}</h4>
                        <p class="text-xs text-[#9CA3AF] truncate">{{ $song->artist }}</p>
                    </div>
                </div>

                {{-- Tombol Hapus (Uncheck) --}}
                <button type="button" wire:click="removeSong({{ $index }})"
                    class="w-8 h-8 flex items-center justify-center rounded-lg text-[#9CA3AF] hover:text-secondary-angry-100 hover:bg-secondary-angry-100/10 transition-all">
                    <i class="fa-solid fa-xmark"></i>
                </button>

                {{-- PENTING: Input Hidden untuk Submit Form Utama --}}
                {{-- Ini yang akan dibaca oleh Request di Controller saat tombol Save ditekan --}}
                <input type="hidden" name="songs[]" value="{{ $song->id }}">
            </div>
        @endforeach
    @endif

</div>
