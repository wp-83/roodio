@extends('layouts.admin.master')

@section('title', 'Playlists')
@section('page_title', 'Playlists')
@section('page_subtitle', 'Manage music collections')

@section('content')

{{--
    PERBAIKAN 1: x-data kita rapikan.
    Kita buat fungsi 'confirmDelete' agar logika tidak menumpuk di HTML.
--}}
<div class="w-full"
     x-data="{
        showDeleteModal: false,
        deleteUrl: '',
        deleteName: '',
        confirmDelete(url, name) {
            this.deleteUrl = url;
            this.deleteName = name;
            this.showDeleteModal = true;
        }
     }">

    {{-- HEADER --}}
    <div class="flex flex-col md:flex-row justify-between items-end mb-6 gap-6">
        <div>
            <h1 class="font-primary text-title text-white font-bold tracking-tight">Playlists Library</h1>
            <p class="font-secondaryAndButton text-body-size text-shadedOfGray-30 mt-1">Curated collections and user-generated playlists.</p>
        </div>
        <a href="{{ route('admin.playlists.create') }}" class="group bg-secondary-happy-100 hover:bg-secondary-happy-85 text-white font-secondaryAndButton font-bold px-6 py-3 rounded-xl shadow-lg shadow-secondary-happy-100/20 transition-all duration-200 transform hover:-translate-y-0.5 flex items-center gap-3 border border-secondary-happy-100/50">
            <span class="text-xl leading-none">+</span>
            <span>Create New Playlist</span>
        </a>
    </div>

    {{-- SEARCH & FILTER BAR SECTION --}}
    <div class="mb-8">
        <form action="{{ route('admin.playlists.index') }}" method="GET" class="flex flex-col xl:flex-row gap-4 justify-between items-center w-full">

            {{-- 1. Search Input --}}
            <div class="relative w-full group">
                <span class="absolute inset-y-0 left-0 flex items-center pl-4 text-primary-20 group-focus-within:text-secondary-happy-100 transition-colors">
                    <i class="fa-solid fa-magnifying-glass"></i>
                </span>
                <input type="text" name="search" value="{{ request('search') }}"
                       class="w-full bg-primary-85 border border-primary-70 rounded-xl py-3 pl-11 pr-12 text-white placeholder-primary-20 focus:outline-none focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 transition-all shadow-md text-sm font-secondaryAndButton"
                       placeholder="Search playlist name..."
                       onkeydown="if(event.key === 'Enter'){ this.form.submit(); }">

                @if(request('search'))
                    <a href="{{ route('admin.playlists.index', request()->except('search')) }}" class="absolute inset-y-0 right-0 flex items-center pr-4 text-primary-20 hover:text-white transition-colors">
                        <i class="fa-solid fa-xmark"></i>
                    </a>
                @endif
            </div>

            {{-- 2. Filter Group --}}
            <div class="flex flex-col sm:flex-row gap-3 w-full xl:w-auto flex-shrink-0">
                {{-- Status Filter --}}
                <div class="relative w-full sm:w-48">
                    <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-primary-20 pointer-events-none">
                        <i class="fa-solid fa-music text-xs"></i>
                    </span>
                    <select name="status" onchange="this.form.submit()" class="w-full appearance-none bg-primary-85 border border-primary-70 text-white text-sm rounded-xl py-3 pl-9 pr-8 focus:outline-none focus:border-secondary-happy-100 cursor-pointer hover:border-primary-60 transition-colors shadow-md font-secondaryAndButton">
                        <option value="">All Status</option>
                        <option value="not_empty" {{ request('status') == 'not_empty' ? 'selected' : '' }}>Has Songs</option>
                        <option value="empty" {{ request('status') == 'empty' ? 'selected' : '' }}>Empty (0 Songs)</option>
                    </select>
                    <div class="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none text-shadedOfGray-50">
                        <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </div>
                </div>

                {{-- Date Filter --}}
                <div class="relative w-full sm:w-44">
                    <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-primary-20 pointer-events-none">
                        <i class="fa-regular fa-calendar text-xs"></i>
                    </span>
                    <select name="sort" onchange="this.form.submit()" class="w-full appearance-none bg-primary-85 border border-primary-70 text-white text-sm rounded-xl py-3 pl-9 pr-8 focus:outline-none focus:border-secondary-happy-100 cursor-pointer hover:border-primary-60 transition-colors shadow-md font-secondaryAndButton">
                        <option value="newest" {{ request('sort') == 'newest' ? 'selected' : '' }}>Newest First</option>
                        <option value="oldest" {{ request('sort') == 'oldest' ? 'selected' : '' }}>Oldest First</option>
                    </select>
                    <div class="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none text-shadedOfGray-50">
                        <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </div>
                </div>

                {{-- Reset Button --}}
                @if(request('search') || request('status') || request('sort'))
                    <a href="{{ route('admin.playlists.index') }}" class="h-[46px] w-[46px] flex items-center justify-center bg-secondary-angry-100/10 hover:bg-secondary-angry-100/20 text-secondary-angry-100 border border-secondary-angry-100/30 rounded-xl transition-all shadow-md flex-shrink-0" title="Reset Filters">
                        <i class="fa-solid fa-rotate-left"></i>
                    </a>
                @endif
            </div>
        </form>

        {{-- Info Result Count --}}
        @if(request('search') || request('status') || request('sort') == 'oldest')
            <div class="mt-3 text-xs text-primary-20 font-secondaryAndButton px-1 flex items-center gap-1">
                <i class="fa-solid fa-filter text-[10px]"></i>
                <span>Found <span class="text-white font-bold">{{ $playlists->total() }}</span> playlists</span>
            </div>
        @endif
    </div>

    {{-- FLASH MESSAGE --}}
    @if(session('success'))
        <div x-data="{ show: true }"
             x-show="show"
             x-transition.opacity.duration.300ms
             class="mb-8 bg-[#0d1f67] border border-secondary-relaxed-100 text-secondary-relaxed-100 px-5 py-4 rounded-xl relative shadow-lg flex items-center gap-3 animate-fade-in-down pr-12">

            <i class="fa-solid fa-circle-check text-xl"></i>
            <span class="block sm:inline font-medium">{{ session('success') }}</span>

            {{-- Close Button --}}
            <button @click="show = false" class="absolute top-0 bottom-0 right-0 px-4 py-3 hover:text-white transition-colors focus:outline-none">
                <i class="fa-solid fa-xmark text-lg"></i>
            </button>
        </div>
    @endif

    {{-- WRAPPER GRID --}}
    <div class="relative w-full group">

        {{-- Previous Page Button --}}
        @if (!$playlists->onFirstPage())
            <a href="{{ $playlists->appends(request()->query())->previousPageUrl() }}"
               class="absolute -left-4 top-1/2 -translate-y-1/2 z-30 w-12 h-12 bg-primary-100/90 backdrop-blur-sm border border-primary-60 text-white rounded-full flex items-center justify-center hover:bg-secondary-happy-100 hover:border-secondary-happy-100 shadow-2xl transition-all duration-300 group-hover:scale-110">
                <i class="fa-solid fa-chevron-left"></i>
            </a>
        @endif

        {{-- GRID LAYOUT --}}
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 animate-slide-up">
            @forelse($playlists as $playlist)
                {{-- CARD --}}
                <div class="group/card bg-primary-85 rounded-xl border border-primary-70 shadow-lg hover:shadow-2xl hover:border-secondary-happy-100/50 transition-all duration-300 flex flex-col h-full relative overflow-hidden">

                    {{-- Cover Art --}}
                    <div class="relative w-full aspect-[16/9] bg-[#020a36] overflow-hidden">
                        @if($playlist->playlistPath)
                            <div class="absolute inset-0 flex items-center justify-center text-primary-60 group-hover/card:scale-110 transition-transform duration-500">
                                <img src="{{ config('filesystems.storage_url') . '/' . $playlist->playlistPath }}"
                                     class="w-full h-full object-cover"
                                     onerror="this.onerror=null; this.parentElement.innerHTML='<i class=\'fa-solid fa-music text-4xl opacity-50\'></i>';">
                            </div>
                        @else
                            <div class="absolute inset-0 flex items-center justify-center text-primary-60 group-hover/card:scale-110 transition-transform duration-500">
                                <i class="fa-solid fa-music text-4xl opacity-50"></i>
                            </div>
                        @endif
                        <div class="absolute inset-0 bg-gradient-to-t from-primary-85 via-transparent to-transparent opacity-90"></div>

                        {{-- Edit Button (Floating) --}}
                        <div class="absolute top-2 right-2 opacity-0 group-hover/card:opacity-100 transition-opacity duration-300 z-20">
                            <a href="{{ route('admin.playlists.edit', $playlist) }}"
                               class="w-8 h-8 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center text-white hover:bg-white hover:text-primary-100 shadow-lg transition-all border border-white/20">
                                <i class="fa-solid fa-pen text-[10px]"></i>
                            </a>
                        </div>
                    </div>

                    {{-- Body --}}
                    <div class="p-4 flex flex-col flex-grow relative -mt-8 z-10">
                        <h3 class="font-primary text-base font-bold text-white mb-0.5 line-clamp-1 group-hover/card:text-secondary-happy-100 transition-colors" title="{{ $playlist->name }}">
                            {{ $playlist->name }}
                        </h3>

                        <div class="flex items-center justify-between text-[10px] text-shadedOfGray-30 mb-2">
                            <span class="flex items-center gap-1 truncate max-w-[100px]">
                                <i class="fa-solid fa-user-circle"></i>
                                {{ $playlist->user->username ?? 'System' }}
                            </span>
                            <span class="px-1.5 py-0.5 rounded bg-primary-70 border border-primary-60 text-white font-mono">
                                {{ $playlist->songs_count ?? 0 }} Songs
                            </span>
                        </div>

                        <p class="font-secondaryAndButton text-xs text-primary-20 line-clamp-2 leading-snug mb-3 flex-grow h-8">
                            {{ $playlist->description ?: 'No description.' }}
                        </p>

                        <div class="pt-2 border-t border-primary-70 flex justify-between items-center mt-auto">
                            <a href="{{ route('admin.playlists.edit', $playlist) }}"
                               class="text-[10px] font-bold text-accent-100 hover:text-white transition-colors flex items-center gap-1">
                                DETAILS <i class="fa-solid fa-arrow-right text-[8px]"></i>
                            </a>

                            {{--
                                PERBAIKAN 2: TOMBOL DELETE
                                - Menggunakan @click.prevent agar tidak reload
                                - Menggunakan fungsi 'confirmDelete' yang didefinisikan di x-data
                                - addslashes() agar string aman dari error javascript
                            --}}
                            <button type="button"
                                    @click.prevent="confirmDelete('{{ route('admin.playlists.destroy', $playlist) }}', '{{ addslashes($playlist->name) }}')"
                                    class="text-shadedOfGray-50 hover:text-secondary-angry-100 transition-colors text-sm px-2 py-1"
                                    title="Delete">
                                <i class="fa-regular fa-trash-can"></i>
                            </button>
                        </div>
                    </div>
                </div>
            @empty
                <div class="col-span-full py-16 flex flex-col items-center justify-center text-center border-2 border-dashed border-primary-70 rounded-2xl bg-primary-85/30">
                    <div class="w-16 h-16 bg-primary-70/50 rounded-full flex items-center justify-center mb-4 text-[#9CA3AF]">
                        <i class="fa-solid fa-folder-open text-3xl"></i>
                    </div>
                    <h3 class="font-primary text-lg text-white font-bold">No Playlists Found</h3>
                    <p class="text-sm text-shadedOfGray-30 mb-4">Start by creating your first collection.</p>
                </div>
            @endforelse
        </div>

        {{-- Next Page Button --}}
        @if ($playlists->hasMorePages())
            <a href="{{ $playlists->appends(request()->query())->nextPageUrl() }}"
               class="absolute -right-4 top-1/2 -translate-y-1/2 z-30 w-12 h-12 bg-primary-100/90 backdrop-blur-sm border border-primary-60 text-white rounded-full flex items-center justify-center hover:bg-secondary-happy-100 hover:border-secondary-happy-100 shadow-2xl transition-all duration-300 group-hover:scale-110">
                <i class="fa-solid fa-chevron-right"></i>
            </a>
        @endif
    </div>

    {{-- 4. PAGINATION INFO --}}
    @if($playlists->hasPages())
        <div class="mt-8 p-4 border-t border-primary-70 bg-primary-85/50 rounded-xl hidden-arrows">
            {{ $playlists->appends(request()->query())->links('pagination.admin') }}
        </div>
    @endif

    <template x-teleport="body">
    {{--
        PERBAIKAN POSISI:
        - lg:left-72 : Geser container modal ke kanan selebar sidebar
        - lg:top-20  : Geser container modal ke bawah setinggi navbar
        - z-50       : Cukup z-50 (tidak perlu 9999) karena area sidebar sudah dihindari
        - flex justify-center items-center : Agar modal panel ada di tengah area tersebut
    --}}
    <div x-show="showDeleteModal" style="display: none;"
         class="fixed inset-0 lg:left-72 lg:top-20 z-50 flex items-center justify-center"
         aria-labelledby="modal-title" role="dialog" aria-modal="true">

        {{-- Backdrop (Hanya menggelapkan area konten) --}}
        <div class="absolute inset-0 bg-[#020a36]/80 backdrop-blur-sm transition-opacity"
             @click="showDeleteModal = false"
             x-show="showDeleteModal"
             x-transition:enter="ease-out duration-300" x-transition:enter-start="opacity-0" x-transition:enter-end="opacity-100"
             x-transition:leave="ease-in duration-200" x-transition:leave-start="opacity-100" x-transition:leave-end="opacity-0">
        </div>

        {{-- Modal Panel --}}
        <div class="relative w-full max-w-md m-4 bg-primary-85 rounded-2xl text-left overflow-hidden shadow-2xl border border-primary-70 transform transition-all"
             x-show="showDeleteModal"
             x-transition:enter="ease-out duration-300"
             x-transition:enter-start="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
             x-transition:enter-end="opacity-100 translate-y-0 sm:scale-100"
             x-transition:leave="ease-in duration-200"
             x-transition:leave-start="opacity-100 translate-y-0 sm:scale-100"
             x-transition:leave-end="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95">

            <div class="p-6">
                <div class="flex items-center gap-4 mb-4">
                    <div class="w-12 h-12 rounded-full bg-secondary-angry-100/20 flex items-center justify-center flex-shrink-0 text-secondary-angry-100 text-xl border border-secondary-angry-100/30 shadow-inner">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                    </div>
                    <div>
                        <h4 class="font-primary font-bold text-white text-lg">Delete Playlist?</h4>
                        <p class="font-secondaryAndButton text-sm text-shadedOfGray-30 mt-1">
                            Are you sure you want to delete <span x-text="deleteName" class="text-white font-bold"></span>? This action cannot be undone.
                        </p>
                    </div>
                </div>

                {{-- Form Action Dinamis --}}
                <form :action="deleteUrl" method="POST">
                    @csrf
                    @method('DELETE')

                    <div class="flex justify-end gap-3 mt-6 pt-4 border-t border-primary-70">
                        <button type="button" @click="showDeleteModal = false"
                                class="px-5 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-30 text-sm font-bold hover:bg-primary-70 hover:text-white transition-colors duration-200">
                            Cancel
                        </button>

                        <button type="submit"
                                class="px-5 py-2.5 rounded-xl bg-secondary-angry-100 text-white text-sm font-bold hover:bg-secondary-angry-85 shadow-lg shadow-secondary-angry-100/20 transition-all duration-200 flex items-center gap-2 transform active:scale-95">
                            <i class="fa-solid fa-trash"></i> Yes, Delete
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</template>

</div>

<style>
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-slide-up {
        animation: slideUpFade 0.4s ease-out forwards;
    }
    .hidden-arrows a[rel="prev"],
    .hidden-arrows a[rel="next"],
    .hidden-arrows span[aria-label*="Previous"],
    .hidden-arrows span[aria-label*="Next"] {
        display: none !important;
    }
</style>
@endsection
