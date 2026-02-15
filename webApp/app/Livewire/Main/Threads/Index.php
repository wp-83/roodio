<?php

namespace App\Livewire\Main\Threads;

use App\Models\Thread;
use Illuminate\Support\Facades\Auth;
use Livewire\Component;
use Livewire\Attributes\Url;
use Livewire\Attributes\Validate;

use Livewire\Attributes\On;

class Index extends Component
{
    #[On('update-search')]
    public function updateSearch($query)
    {
        $this->search = $query;
    }

    #[Url]
    public $filter = 'all';

    #[Url]
    public $search = '';

    public $mood;

    // Create Thread Properties
    #[Validate('required|max:100')]
    public $title = '';

    #[Validate('required|max:255')]
    public $content = '';

    public $isReplyable = false;

    public function mount()
    {
        $this->mood = session('chooseMood', 'happy');
    }

    public function saveThread()
    {
        $this->validate();

        Thread::create([
            'title' => $this->title,
            'content' => $this->content,
            'userId' => Auth::id(),
            'datePost' => now(),
            'isReplyable' => $this->isReplyable,
        ]);

        $this->reset(['title', 'content', 'isReplyable']);
        $this->dispatch('close-modal', 'createThreadPopup'); // Assuming you have a JS listener for this
        $this->dispatch('thread-created'); 
    }

    public function render()
    {
        $user = Auth::user();

        $query = Thread::withCount('reactions')->orderByDesc('created_at');

        // Apply search if provided
        if (!empty($this->search)) {
            $query->search($this->search);
        }

        if ($this->filter == 'created') {
            $query->where('userId', $user->id);
        } else if ($this->filter == 'following') {
            $followingIds = $user->followings()->pluck('users.id');
            $query->whereIn('userId', $followingIds);
        }

        $threads = $query->get();
        $fullname = $user->userDetail->fullname;

        return view('livewire.main.threads.index', [
            'threads' => $threads,
            'user' => $user,
            'fullname' => $fullname
        ])->layout('layouts.main', ['mood' => $this->mood]);
    }
}
