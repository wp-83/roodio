@extends('layouts.admin.master')

@section('title', 'Songs Library')
@section('page_title', 'Songs Library')
@section('page_subtitle', 'Manage your music library catalogue')

@section('content')
<div class="w-full">

    {{-- Header Section --}}
    <div class="flex flex-col md:flex-row justify-between items-end mb-8 gap-4">
        <div>
            {{-- Menggunakan text-white untuk kontras maksimal di bg primary-100 --}}
            <h1 class="font-primary text-title text-white font-bold tracking-tight">Songs Management</h1>
            <p class="font-secondaryAndButton text-body-size text-shadedOfGray-30 mt-1">Manage and organize your entire music library catalogue.</p>
        </div>

        {{-- Button menggunakan Secondary Happy (Orange) agar 'pop' di background biru gelap --}}
        <a href="{{ route('admin.songs.create') }}" class="group bg-secondary-happy-100 hover:bg-secondary-happy-85 text-white font-secondaryAndButton font-medium px-6 py-3 rounded-xl shadow-lg shadow-secondary-happy-100/20 transition-all duration-200 transform hover:-translate-y-0.5 flex items-center gap-3 border border-secondary-happy-100/50">
            <span class="text-lg leading-none">+</span>
            <span>Add New Song</span>
        </a>
    </div>

    {{-- Content Card --}}
    <div class="bg-primary-85 rounded-2xl shadow-xl border border-primary-70 overflow-hidden">
        {{-- Wrapper tetap overflow-x-auto untuk safety di Mobile, tapi di Desktop akan fit --}}
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">

                {{-- Table Header --}}
                <thead class="bg-primary-70 text-shadedOfGray-20 font-primary border-b border-primary-60">
                    <tr>
                        {{-- Details: Auto width (mengisi sisa ruang) --}}
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs">Details</th>

                        {{-- Admin: Lebar secukupnya --}}
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs w-32">Admin</th>

                        {{-- Genre: Lebar secukupnya --}}
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs w-24">Genre</th>

                        {{-- Mood: Lebar fix --}}
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs w-32">Mood</th>

                        {{-- Meta: Lebar fix --}}
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs w-40">Meta</th>

                        {{-- Actions: Lebar fix --}}
                        <th class="px-4 py-5 font-bold text-small uppercase tracking-wider text-xs text-center w-32">Actions</th>
                    </tr>
                </thead>

                {{-- Table Body --}}
                <tbody class="font-secondaryAndButton text-small text-white divide-y divide-primary-70">
                    @forelse($songs as $song)
                        <tr class="hover:bg-primary-70/40 transition duration-200 group">
                            {{-- Title & Artist (Truncate agar tidak melebar) --}}
                            <td class="px-4 py-4 max-w-xs xl:max-w-sm">
                                <div class="flex flex-col">
                                    <span class="font-bold text-white text-body-size mb-0.5 truncate block group-hover:text-secondary-happy-100 transition-colors" title="{{ $song->title }}">
                                        {{ $song->title }}
                                    </span>
                                    <div class="flex items-center gap-2 text-shadedOfGray-30 text-xs">
                                        <span class="font-medium text-shadedOfGray-20 truncate max-w-[150px]">{{ $song->artist }}</span>
                                        <span class="text-shadedOfGray-50">•</span>
                                        <span class="truncate max-w-[100px]">{{ $song->publisher }}</span>
                                    </div>
                                </div>
                            </td>

                            {{-- Admin Badge --}}
                            <td class="px-4 py-4">
                                @if($song->user() && $song->user)
                                    <div class="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-primary-60 bg-primary-100/50 max-w-full">
                                        <div class="w-1.5 h-1.5 rounded-full bg-secondary-happy-100 flex-shrink-0"></div>
                                        <span class="text-xs font-medium text-shadedOfGray-10 truncate">{{ $song->user->username }}</span>
                                    </div>
                                @else
                                    <span class="text-shadedOfGray-50 italic text-xs bg-primary-70/30 px-2 py-1 rounded">System</span>
                                @endif
                            </td>

                            {{-- Genre Badge --}}
                            <td class="px-4 py-4">
                                <span class="inline-flex items-center px-2.5 py-1 rounded-lg bg-accent-100/10 text-accent-100 border border-accent-100/20 text-xs font-bold tracking-wide shadow-sm truncate max-w-full">
                                    {{ $song->genre }}
                                </span>
                            </td>

                            {{-- Mood Badge (New Design) --}}
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
                                        'happy'   => 'bi-emoji-smile',
                                        'sad'     => 'bi-emoji-frown',
                                        'relaxed' => 'bi-cup-hot',
                                        'angry'   => 'bi-fire',
                                        default   => 'bi-music-note',
                                    };
                                @endphp
                                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border {{ $moodColor }} text-xs font-bold tracking-wide capitalize shadow-sm w-fit max-w-full truncate">
                                    <i class="bi {{ $moodIcon }}"></i>
                                    <span class="truncate">{{ $song->mood->type ?? 'Unknown' }}</span>
                                </span>
                            </td>

                            {{-- Meta --}}
                            <td class="px-4 py-4">
                                <div class="flex flex-col gap-1">
                                    <div class="flex items-center gap-1.5 text-shadedOfGray-10 text-xs font-medium">
                                        <i class="bi bi-clock text-secondary-happy-85"></i> {{ $song->duration }}
                                    </div>
                                    <div class="text-shadedOfGray-50 text-[10px] truncate">
                                        {{ $song->datePublished }}
                                    </div>
                                </div>
                            </td>

                            {{-- Actions --}}
                            <td class="px-4 py-4 text-center">
                                <div class="flex items-center justify-center gap-2 opacity-80 group-hover:opacity-100 transition-opacity">
                                    <a href="{{ route('admin.songs.edit', $song) }}"
                                    class="w-8 h-8 flex items-center justify-center rounded-lg bg-primary-60 text-white hover:bg-white hover:text-primary-85 transition-all duration-200 shadow-sm border border-primary-50"
                                    title="Update Song">
                                        <i class="bi bi-pencil-fill text-xs"></i>
                                    </a>

                                    <form action="{{ route('admin.songs.destroy', $song) }}" method="POST" class="inline-block">
                                        @csrf
                                        @method('DELETE')
                                        <button type="button"
                                                onclick="openDeleteMusic('{{ route('admin.songs.destroy', $song) }}', '{{ $song->title }}')"
                                                class="w-8 h-8 flex items-center justify-center rounded-lg bg-error-dark/20 text-error-moderate hover:bg-error-moderate hover:text-white transition-all duration-200 shadow-sm border border-error-dark/30"
                                                title="Delete">
                                            <i class="bi bi-trash-fill text-xs"></i>
                                        </button>
                                    </form>
                                </div>
                            </td>
                        </tr>
                    @empty
                        {{-- Empty State --}}
                        <tr>
                            <td colspan="7" class="py-24 text-center">
                                <div class="flex flex-col items-center justify-center">
                                    <div class="w-24 h-24 bg-primary-70/50 rounded-full flex items-center justify-center mb-6 text-shadedOfGray-40 border border-primary-60 animate-pulse">
                                        <i class="bi bi-music-note-list text-4xl"></i>
                                    </div>
                                    <h3 class="font-primary text-subtitle text-white font-bold mb-2">Library is Empty</h3>
                                    <p class="font-secondaryAndButton text-shadedOfGray-30 max-w-sm mx-auto mb-8">
                                        Your playlist is currently empty. Start building your music collection now.
                                    </p>
                                    <a href="{{ route('admin.songs.create') }}" class="px-6 py-3 rounded-xl bg-primary-60 text-white font-bold hover:bg-primary-50 transition border border-primary-50 flex items-center gap-2">
                                        <i class="bi bi-plus-lg"></i> Add First Song
                                    </a>
                                </div>
                            </td>
                        </tr>
                    @endforelse
                </tbody>
            </table>
        </div>
    </div>

    {{-- DELETE CONFIRMATION MODAL --}}
    <div id="deleteMusicModal" class="hidden fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
        <div class="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            {{-- Backdrop --}}
            <div class="fixed inset-0 bg-[#020a36]/80 backdrop-blur-sm transition-opacity" onclick="toggleDeleteMusicModal()"></div>
            <span class="hidden sm:inline-block sm:align-middle sm:h-screen">&#8203;</span>

            {{-- Modal Panel --}}
            <div class="relative inline-block align-bottom bg-primary-85 rounded-2xl text-left overflow-hidden shadow-2xl transform transition-all sm:my-8 sm:align-middle sm:max-w-md w-full border border-primary-70">
                <div class="p-6">
                    <div class="flex items-center gap-4 mb-4">
                        {{-- Warning Icon --}}
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
                            {{-- Cancel Button --}}
                            <button type="button" onclick="toggleDeleteMusicModal()"
                                    class="px-5 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-30 text-sm font-bold hover:bg-primary-70 hover:text-white transition-colors duration-200">
                                Cancel
                            </button>

                            {{-- Delete Button (Styled) --}}
                            <button type="submit"
                                    class="px-5 py-2.5 rounded-xl bg-secondary-angry-100 text-white text-sm font-bold hover:bg-secondary-angry-85 shadow-lg shadow-secondary-angry-100/20 transition-all duration-200 flex items-center gap-2 transform active:scale-95">
                                <i class="bi bi-trash"></i> Yes, Delete
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    // 1. DELETE LOGIC
    function toggleDeleteMusicModal() {
        const modal = document.getElementById('deleteMusicModal');

        if (modal.classList.contains('hidden')) {
            modal.classList.remove('hidden');
            // Animasi Masuk (Optional: pakai Tailwind animate class jika ada)
            // document.body.style.overflow = 'hidden';
        } else {
            modal.classList.add('hidden');
            // document.body.style.overflow = 'auto';
        }
    }

    // Fungsi ini dipanggil dari tombol trash di tabel (ubah onclick di tabel)
    // Contoh: onclick="openDeleteMusic('{{ route('admin.songs.destroy', $song) }}', '{{ $song->title }}')"
    function openDeleteMusic(actionUrl, itemName) {
        const form = document.getElementById('deleteMusicForm');
        form.action = actionUrl;
        document.getElementById('delete_item_name').textContent = `"${itemName}"`;
        toggleDeleteMusicModal();
    }
</script>
@endsection
