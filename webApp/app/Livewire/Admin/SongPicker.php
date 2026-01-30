<?php
namespace App\Livewire\Admin;

use App\Models\Songs;
use Livewire\Attributes\On;
use Livewire\Component;

// Import Class Penerima

class SongPicker extends Component
{
    public $search          = '';
    public $selectedSongIds = [];

    public function selectSong($songId)
    {
        $this->dispatch('song-added', $songId)->to(PlaylistSelectedSongs::class);

        if (! in_array($songId, $this->selectedSongIds)) {
            $this->selectedSongIds[] = $songId;
        }
    }

    #[On('song-removed')]
    public function restoreSongState($songId)
    {
        $this->selectedSongIds = array_values(array_diff($this->selectedSongIds, [$songId]));
    }

    public function render()
    {
        $songs = Songs::query()
            ->when($this->search, function ($query) {
                $query->where('title', 'like', '%' . $this->search . '%')
                    ->orWhere('artist', 'like', '%' . $this->search . '%');
            })
            ->whereNotIn('id', $this->selectedSongIds)
            ->get();

        return view('livewire.admin.songPicker', compact('songs'));
    }
}
