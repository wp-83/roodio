@extends('layouts.main')

@section('mainContentContainerClass')

@php
    $mood = 'relaxed';
@endphp

{{-- 1. Definisikan Data Playlist DULU --}}
@php
    $playlistData = $songs->map(function($song) {
        return [
            'id' => $song->id,
            'title' => $song->title,
            'artist' => $song->artist,
            'cover' => $song->cover_path ?? asset('assets/images/default-music.png'), // Pastikan ada default image
            'path' => config('filesystems.disks.azure.url') . '/' . $song->songPath
        ];
    });
@endphp

<script>
    // Taruh data ke window global
    window.currentPlaylist = @json($playlistData);
</script>

{{-- 2. BARU panggil script audioControl.js SETELAH data tersedia --}}
@push('script')
    <script src="{{ asset('js/pages/audioControl.js') }}" defer></script>
@endpush

@section('mainContent')
    <div class="flex flex-col gap-4">
        @foreach ($songs as $index => $song)
            <div class="flex items-center justify-between p-3 bg-white rounded shadow">
                <div>
                    <p class="font-bold">{{ $song->title }}</p>
                    <p class="text-sm text-gray-500">{{ $song->artist }}</p>
                </div>

                {{-- Tombol Play di List --}}
                {{-- Kita panggil fungsi playSongByIndex() dengan urutan array --}}
                <button onclick="playSongByIndex({{ $index }})"
                        class="px-4 py-2 text-white rounded bg-secondary-happy-85 hover:bg-secondary-happy-70">
                    Play
                </button>
            </div>
        @endforeach
    </div>
@endsection

@section('bottomContent')
    <x-audioPlayer :mood="$mood"></x-audioPlayer>
@endsection
