@extends('layouts.master')

@section('title', 'Playlists Admin')

@section('bodyContent')
<div class="w-full px-6 py-10">

    <div class="flex flex-col md:flex-row justify-between items-center mb-10">
        <div>
            <h1 class="font-primary text-title text-white font-bold">Songs Management</h1>
            <p class="font-secondaryAndButton text-body-size text-shadedOfGray-20">Manage your music library catalogue</p>
        </div>
        <a href="{{ route('admin.songs.create') }}" class="mt-4 md:mt-0 bg-secondary-happy-100 hover:bg-secondary-happy-85 text-white font-secondaryAndButton font-medium px-6 py-3 rounded-lg shadow-md transition duration-200 flex items-center gap-2">
            <span>+ Add New Song</span>
        </a>
    </div>

    <div class="bg-white rounded-xl shadow-lg border border-shadedOfGray-20 overflow-hidden">
        <div class="overflow-x-auto">
            <table class="w-full min-w-[1000px] text-left border-collapse">
                <thead class="bg-primary-85 text-white font-primary">
                    <tr>
                        <th class="p-4 font-semibold text-small">ID</th>
                        <th class="p-4 font-semibold text-small">Admin</th>
                        <th class="p-4 font-semibold text-small">Title</th>
                        <th class="p-4 font-semibold text-small w-48">Lyrics Preview</th>
                        <th class="p-4 font-semibold text-small">Artist</th>
                        <th class="p-4 font-semibold text-small">Genre</th>
                        <th class="p-4 font-semibold text-small">Duration</th>
                        <th class="p-4 font-semibold text-small">Publisher</th>
                        <th class="p-4 font-semibold text-small">Date</th>
                        <th class="p-4 font-semibold text-small text-center">Actions</th>
                    </tr>
                </thead>
                <tbody class="font-secondaryAndButton text-small text-primary-85">
                    @forelse($songs as $song)
                        <tr class="border-b border-shadedOfGray-10 hover:bg-shadedOfGray-10 transition duration-150 odd:bg-white even:bg-white">
                            <td class="p-4 text-shadedOfGray-60">#{{ $song->id }}</td>

                            <td class="p-4">
                                @if($song->user() && $song->user)
                                    {{ $song->user->username}}
                                @else
                                    <span class="text-shadedOfGray-40 italic">System/Unassigned</span>
                                @endif
                            </td>

                            <td class="p-4 font-medium text-primary-70">{{ $song->title }}</td>
                            <td class="p-4 truncate max-w-xs text-shadedOfGray-60" title="{{ $song->lyrics }}">
                                {{ Str::limit($song->lyrics, 30) }}
                            </td>
                            <td class="p-4">{{ $song->artist }}</td>
                            <td class="p-4">
                                <span class="px-3 py-1 rounded-full bg-secondary-relaxed-20 text-secondary-relaxed-100 text-micro font-bold">
                                    {{ $song->genre }}
                                </span>
                            </td>
                            <td class="p-4">{{ $song->duration }}</td>
                            <td class="p-4">{{ $song->publisher }}</td>
                            <td class="p-4 whitespace-nowrap">{{ $song->datePublished }}</td>

                            <td class="p-4">
                                <div class="flex items-center justify-center gap-2">
                                    <a href="{{ route('admin.songs.edit', $song) }}" class="p-2 text-primary-50 hover:text-primary-60 hover:bg-primary-10 rounded-lg transition" title="Edit">
                                        Update
                                    </a>

                                    <form action="{{ route('admin.songs.destroy', $song) }}" method="POST" class="inline">
                                        @csrf
                                        @method('DELETE')
                                        <button type="submit" onclick="return confirm('Are You Sure to delete?')" class="p-2 text-error-moderate hover:text-error-dark hover:bg-error-lighten/20 rounded-lg transition" title="Delete">
                                            Delete
                                        </button>
                                    </form>
                                </div>
                            </td>
                        </tr>
                    @empty
                        <tr>
                            <td colspan="10" class="p-12 text-center">
                                <div class="flex flex-col items-center justify-center text-shadedOfGray-40">
                                    <p class="font-primary text-subtitle mb-2">Yeay santai dulu!</p>
                                    <p class="font-secondaryAndButton">Playlist kosong, belum ada lagu yang ditambahkan.</p>
                                </div>
                            </td>
                        </tr>
                    @endforelse
                </tbody>
            </table>
        </div>
    </div>

    <div class="mt-8 pt-6 border-t border-shadedOfGray-30/30">
        <form action="{{ route('auth.logout') }}" method="post">
            @csrf
            <button type="submit" class="group flex items-center gap-2 text-error-moderate hover:text-error-lighten font-secondaryAndButton font-medium transition pl-1">
                <i class="bi bi-door-open group-hover:-translate-x-1 transition-transform"></i>
                Logout
            </button>
        </form>
    </div>
</div>
@endsection
