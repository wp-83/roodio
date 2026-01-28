<?php
namespace App\Livewire\Admin;

use App\Models\Songs;
use Livewire\Attributes\On;
use Livewire\Component;

class PlaylistSelectedSongs extends Component
{
    public $selectedSongs = [];

    // Listener menangkap parameter pertama dari dispatch
    #[On('song-added')]
    public function addSong($songId)
    {
        // 1. Cek Duplikasi (Cegah lagu yang sama masuk 2x)
        // Kita cek column 'id' di dalam collection/array
        if (collect($this->selectedSongs)->where('id', $songId)->isNotEmpty()) {
            return;
        }

        // 2. Ambil Lagu
        $song = Songs::find($songId);

        if ($song) {
            $this->selectedSongs[] = $song;
        }
    }

    public function removeSong($index)
    {
        $removedSongId = $this->selectedSongs[$index]['id'] ?? $this->selectedSongs[$index]->id;

        unset($this->selectedSongs[$index]);
        $this->selectedSongs = array_values($this->selectedSongs);

        $this->dispatch('song-removed', $removedSongId)->to(SongPicker::class);
    }
    public function mount($initialSongs = [])
    {
        if (! empty($initialSongs)) {
            foreach ($initialSongs as $song) {
                $this->selectedSongs[] = $song;
            }
        }
    }

    public function render()
    {
        return view('livewire.admin.playlistSelectedSongs');
    }
}
