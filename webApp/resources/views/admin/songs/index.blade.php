@extends('layouts.admin.master')

@section('title', 'Songs Library')
@section('page_title', 'Songs Library')
@section('page_subtitle', 'Manage your music library catalogue')

@section('content')
<div class="w-full">

    {{-- FLASH MESSAGE (Tetap sama) --}}
    @if(session('success') || request('success_upload'))
        <div id="flashMessage" class="mb-6 bg-primary-85 border border-secondary-relaxed-100 text-secondary-relaxed-100 px-4 py-3 rounded-xl relative shadow-lg flex items-center gap-3 animate-fade-in-down">
            <i class="bi bi-check-circle-fill text-xl"></i>
            <span class="block sm:inline font-medium font-secondaryAndButton">
                {{ session('success') ?? 'Successfully added song with AI prediction!' }}
            </span>
            <button onclick="document.getElementById('flashMessage').remove()" class="absolute top-0 bottom-0 right-0 px-4 py-3 hover:text-white transition-colors">
                <i class="bi bi-x-lg"></i>
            </button>
        </div>
    @endif

    {{-- PAGE HEADER (Tetap sama) --}}
    <div class="flex flex-col md:flex-row justify-between items-end mb-8 gap-4">
        <div>
            <h1 class="font-primary text-title text-white font-bold tracking-tight">Songs Management</h1>
            <p class="font-secondaryAndButton text-body-size text-shadedOfGray-30 mt-1">Manage and organize your entire music library catalogue.</p>
        </div>
        <a href="{{ route('admin.songs.create') }}" class="group bg-secondary-happy-100 hover:bg-secondary-happy-85 text-white font-secondaryAndButton font-medium px-6 py-3 rounded-xl shadow-lg shadow-secondary-happy-100/20 transition-all duration-200 transform hover:-translate-y-0.5 flex items-center gap-3 border border-secondary-happy-100/50">
            <span class="text-lg leading-none">+</span>
            <span>Add New Song</span>
        </a>
    </div>

    {{-- CONTENT CARD --}}
    <div class="bg-primary-85 rounded-2xl shadow-xl border border-primary-70 overflow-hidden">

        {{-- === NEW: FILTER & SEARCH TOOLBAR === --}}
        <div class="p-6 border-b border-primary-70">
            <form action="{{ route('admin.songs.index') }}" method="GET" class="flex flex-col md:flex-row gap-4 justify-between items-center">

                {{-- Left: Search Bar --}}
                {{-- Nilai search akan tetap terbawa saat dropdown mood berubah karena berada dalam satu form --}}
                <div class="relative w-full md:w-96 group">
                    <span class="absolute inset-y-0 left-0 pl-4 flex items-center text-shadedOfGray-50 group-focus-within:text-secondary-happy-100 transition-colors">
                        <i class="bi bi-search"></i>
                    </span>
                    <input type="text"
                           name="search"
                           value="{{ request('search') }}"
                           class="w-full bg-primary-100 border border-primary-60 text-white text-sm rounded-xl pl-11 pr-4 py-3 focus:outline-none focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 placeholder-primary-20 transition-all shadow-inner"
                           placeholder="Search title, artist, publisher..."
                           onkeydown="if(event.key === 'Enter'){ this.form.submit(); }">
                           {{-- Search otomatis submit saat Enter --}}
                </div>

                {{-- Right: Mood Filter (Auto Submit) --}}
                <div class="flex items-center gap-3 w-full md:w-auto">

                    {{-- Label Optional (supaya user paham) --}}
                    <span class="text-xs text-shadedOfGray-30 font-bold uppercase tracking-wider hidden md:inline">Filter by Mood:</span>

                    <div class="relative w-full md:w-48">
                        {{-- onchange="this.form.submit()" adalah kunci auto-submit --}}
                        <select name="mood"
                                onchange="this.form.submit()"
                                class="w-full appearance-none bg-primary-100 border border-primary-60 text-white text-sm rounded-xl px-4 py-3 pr-10 focus:outline-none focus:border-secondary-happy-100 cursor-pointer shadow-sm hover:border-primary-50 transition-colors">
                            <option value="">All Moods</option>
                            {{-- Pastikan value sesuai dengan database (lowercase/uppercase) --}}
                            <option value="happy" {{ request('mood') == 'happy' ? 'selected' : '' }}>Happy</option>
                            <option value="sad" {{ request('mood') == 'sad' ? 'selected' : '' }}>Sad</option>
                            <option value="relaxed" {{ request('mood') == 'relaxed' ? 'selected' : '' }}>Relaxed</option>
                            <option value="angry" {{ request('mood') == 'angry' ? 'selected' : '' }}>Angry</option>
                        </select>

                        {{-- Custom Arrow Icon --}}
                        <div class="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none text-shadedOfGray-50">
                            <i class="bi bi-chevron-down text-xs"></i>
                        </div>
                    </div>

                    {{-- Reset Button (Hanya muncul jika ada filter aktif) --}}
                    @if(request('search') || request('mood'))
                        <a href="{{ route('admin.songs.index') }}"
                           class="h-[46px] w-[46px] flex items-center justify-center bg-secondary-angry-100/10 hover:bg-secondary-angry-100/20 text-secondary-angry-100 border border-secondary-angry-100/30 rounded-xl transition-all tooltip-trigger"
                           title="Reset Filters">
                            <i class="bi bi-x-lg text-lg"></i>
                        </a>
                    @endif
                </div>
            </form>
        </div>

        {{-- TABLE (Tetap sama, hanya perlu handle kondisi 'Not Found' karena search) --}}
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">
                {{-- ... THEAD ... --}}
                <thead class="bg-primary-70 text-shadedOfGray-20 font-primary border-b border-primary-60">
                    <tr>
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs">Details</th>
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs w-32">Admin</th>
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs w-24">Genre</th>
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs w-32">Mood</th>
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs w-40">Meta</th>
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs text-center w-32">Actions</th>
                    </tr>
                </thead>

                <tbody class="font-secondaryAndButton text-small text-white divide-y divide-primary-70">
                    @forelse($songs as $song)
                        {{-- ... TR CONTENT (Sama seperti sebelumnya) ... --}}
                        <tr class="hover:bg-primary-70/40 transition duration-200 group">
                            {{-- ... (Isi kolom sama persis seperti sebelumnya) ... --}}
                            {{-- Saya ringkas disini agar tidak terlalu panjang, gunakan isi TR sebelumnya --}}
                            <td class="px-4 py-4 max-w-xs xl:max-w-sm">
                                <div class="flex flex-col">
                                    <span class="font-bold text-white text-body-size mb-0.5 truncate block group-hover:text-secondary-happy-100 transition-colors">{{ $song->title }}</span>
                                    <div class="flex items-center gap-2 text-shadedOfGray-30 text-xs">
                                        <span class="font-medium text-shadedOfGray-20 truncate max-w-[150px]">{{ $song->artist }}</span>
                                        <span class="text-shadedOfGray-50">•</span>
                                        <span class="truncate max-w-[100px]">{{ $song->publisher }}</span>
                                    </div>
                                </div>
                            </td>
                            <td class="px-4 py-4">
                                @if($song->user)
                                    <div class="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-primary-60 bg-primary-100/50 max-w-full">
                                        <div class="w-1.5 h-1.5 rounded-full bg-secondary-happy-100 flex-shrink-0"></div>
                                        <span class="text-xs font-medium text-shadedOfGray-10 truncate">{{ $song->user->username }}</span>
                                    </div>
                                @else
                                    <span class="text-shadedOfGray-50 italic text-xs bg-primary-70/30 px-2 py-1 rounded">System</span>
                                @endif
                            </td>
                            <td class="px-4 py-4">
                                <span class="inline-flex items-center px-2.5 py-1 rounded-lg bg-accent-100/10 text-accent-100 border border-accent-100/20 text-xs font-bold tracking-wide shadow-sm truncate max-w-full">{{ $song->genre }}</span>
                            </td>
                            <td class="px-4 py-4">
                                @php
                                    $moodType = strtolower($song->mood->type ?? '');
                                    $moodColor = match($moodType) {
                                        'happy'   => 'bg-secondary-happy-100/10 text-secondary-happy-100 border-secondary-happy-100/20',
                                        'sad'     => 'bg-secondary-sad-100/10 text-secondary-sad-100 border-secondary-sad-100/20',
                                        'relaxed' => 'bg-secondary-relaxed-100/10 text-secondary-relaxed-100 border-secondary-relaxed-100/20',
                                        'angry'   => 'bg-secondary-angry-100/10 text-secondary-angry-100 border-secondary-angry-100/20',
                                        default   => 'bg-primary-60/20 text-shadedOfGray-10 border-primary-60/30',
                                    };
                                    $moodIcon = match($moodType) {
                                        'happy' => 'bi-emoji-smile', 'sad' => 'bi-emoji-frown', 'relaxed' => 'bi-cup-hot', 'angry' => 'bi-fire', default => 'bi-music-note',
                                    };
                                @endphp
                                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border {{ $moodColor }} text-xs font-bold tracking-wide capitalize shadow-sm w-fit max-w-full truncate">
                                    <i class="bi {{ $moodIcon }}"></i>
                                    <span class="truncate">{{ $song->mood->type ?? 'Unknown' }}</span>
                                </span>
                            </td>
                            <td class="px-4 py-4">
                                <div class="flex flex-col gap-1">
                                    <div class="flex items-center gap-1.5 text-shadedOfGray-10 text-xs font-medium"><i class="bi bi-clock text-secondary-happy-85"></i> {{ $song->duration }}</div>
                                    <div class="text-shadedOfGray-50 text-[10px] truncate">{{ $song->datePublished }}</div>
                                </div>
                            </td>
                            <td class="px-4 py-4 text-center">
                                <div class="flex items-center justify-center gap-2 opacity-80 group-hover:opacity-100 transition-opacity">
                                    <a href="{{ route('admin.songs.edit', $song) }}" class="w-8 h-8 flex items-center justify-center rounded-lg bg-primary-60 text-white hover:bg-white hover:text-primary-85 transition-all duration-200 shadow-sm border border-primary-50"><i class="bi bi-pencil-fill text-xs"></i></a>
                                    <button type="button" onclick="openDeleteMusic('{{ route('admin.songs.destroy', $song) }}', '{{ $song->title }}')" class="w-8 h-8 flex items-center justify-center rounded-lg bg-error-dark/20 text-error-moderate hover:bg-error-moderate hover:text-white transition-all duration-200 shadow-sm border border-error-dark/30"><i class="bi bi-trash-fill text-xs"></i></button>
                                </div>
                            </td>
                        </tr>
                    @empty
                        {{-- Empty State (Updated for Search) --}}
                        <tr>
                            <td colspan="7" class="py-24 text-center">
                                <div class="flex flex-col items-center justify-center">
                                    <div class="w-24 h-24 bg-primary-70/50 rounded-full flex items-center justify-center mb-6 text-primary-20 border border-primary-60">
                                        @if(request('search'))
                                            <i class="bi bi-search text-4xl"></i>
                                        @else
                                            <i class="bi bi-music-note-list text-4xl"></i>
                                        @endif
                                    </div>

                                    @if(request('search') || request('genre') || request('mood'))
                                        <h3 class="font-primary text-subtitle text-white font-bold mb-2">No Results Found</h3>
                                        <p class="font-secondaryAndButton text-shadedOfGray-30 max-w-sm mx-auto mb-6">
                                            We couldn't find any songs matching your filters. Try adjusting your search keywords.
                                        </p>
                                        <a href="{{ route('admin.songs.index') }}" class="px-6 py-2.5 rounded-xl border border-primary-60 text-white hover:bg-primary-70 transition font-bold text-sm">
                                            Clear Filters
                                        </a>
                                    @else
                                        <h3 class="font-primary text-subtitle text-white font-bold mb-2">Library is Empty</h3>
                                        <p class="font-secondaryAndButton text-shadedOfGray-30 max-w-sm mx-auto mb-8">
                                            Your playlist is currently empty. Start building your music collection now.
                                        </p>
                                        <a href="{{ route('admin.songs.create') }}" class="px-6 py-3 rounded-xl bg-primary-60 text-white font-bold hover:bg-primary-50 transition border border-primary-50 flex items-center gap-2">
                                            <i class="bi bi-plus-lg"></i> Add First Song
                                        </a>
                                    @endif
                                </div>
                            </td>
                        </tr>
                    @endforelse
                </tbody>
            </table>
        </div>

        {{-- PAGINATION --}}
        @if($songs->hasPages())
            <div class="px-6 py-6 border-t border-primary-70 bg-primary-85">
                {{ $songs->appends(request()->query())->links('pagination.admin') }}
            </div>
        @endif
    </div>

    {{-- DELETE MODAL (Tetap sama) --}}
    {{-- ... (Kode Modal Delete Anda sebelumnya) ... --}}
    <div id="deleteMusicModal" class="hidden fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
        {{-- ... isi modal ... --}}
        {{-- (Pastikan copy paste kode modal dari respon sebelumnya disini) --}}
        <div class="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            <div class="fixed inset-0 bg-[#020a36]/80 backdrop-blur-sm transition-opacity" onclick="toggleDeleteMusicModal()"></div>
            <span class="hidden sm:inline-block sm:align-middle sm:h-screen">&#8203;</span>
            <div class="relative inline-block align-bottom bg-primary-85 rounded-2xl text-left overflow-hidden shadow-2xl transform transition-all sm:my-8 sm:align-middle sm:max-w-md w-full border border-primary-70">
                <div class="p-6">
                    <div class="flex items-center gap-4 mb-4">
                        <div class="w-12 h-12 rounded-full bg-secondary-angry-100/20 flex items-center justify-center flex-shrink-0 text-secondary-angry-100 text-xl border border-secondary-angry-100/30 shadow-inner">
                            <i class="bi bi-exclamation-triangle-fill"></i>
                        </div>
                        <div>
                            <h4 class="font-primary font-bold text-white text-lg">Delete Item?</h4>
                            <p class="font-secondaryAndButton text-sm text-shadedOfGray-30 mt-1">
                                Are you sure you want to delete <span id="delete_item_name" class="text-white font-bold"></span>? This action cannot be undone.
                            </p>
                        </div>
                    </div>
                    <form id="deleteMusicForm" action="#" method="POST">
                        @csrf
                        @method('DELETE')
                        <div class="flex justify-end gap-3 mt-6 pt-4 border-t border-primary-70">
                            <button type="button" onclick="toggleDeleteMusicModal()" class="px-5 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-30 text-sm font-bold hover:bg-primary-70 hover:text-white transition-colors duration-200">Cancel</button>
                            <button type="submit" class="px-5 py-2.5 rounded-xl bg-secondary-angry-100 text-white text-sm font-bold hover:bg-secondary-angry-85 shadow-lg shadow-secondary-angry-100/20 transition-all duration-200 flex items-center gap-2 transform active:scale-95"><i class="bi bi-trash"></i> Yes, Delete</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>

</div>

{{-- JAVASCRIPT (Tetap sama) --}}
<script>
    function toggleDeleteMusicModal() {
        const modal = document.getElementById('deleteMusicModal');
        if (modal.classList.contains('hidden')) {
            modal.classList.remove('hidden');
        } else {
            modal.classList.add('hidden');
        }
    }
    function openDeleteMusic(actionUrl, itemName) {
        const form = document.getElementById('deleteMusicForm');
        form.action = actionUrl;
        document.getElementById('delete_item_name').textContent = `"${itemName}"`;
        toggleDeleteMusicModal();
    }
</script>
@endsection
