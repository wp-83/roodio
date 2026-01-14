<?php
namespace App\Livewire\Admin;

use App\Models\Songs;
use Livewire\Component;

class SongPicker extends Component
{
    public $search        = '';
    public $selectedSongs = [];

    public function mount($currentSongs = [])
    {
        $this->selectedSongs = $currentSongs;
    }

    public function render()
    {
        $songs = Songs::query()
            ->when($this->search, function ($query) {
                $query->where('title', 'like', '%' . $this->search . '%')
                    ->orWhere('artist', 'like', '%' . $this->search . '%');
            })
            ->limit(10)
            ->get();

        return view('livewire.admin.songPicker', compact('songs'));
    }
}
