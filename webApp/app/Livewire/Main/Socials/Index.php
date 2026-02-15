<?php

namespace App\Livewire\Main\Socials;

use App\Models\User;
use Livewire\Component;
use Livewire\Attributes\Url;
use Livewire\Attributes\On;

class Index extends Component
{
    #[Url]
    public $filter = 'all';

    #[Url]
    public $search = '';

    public $mood;

    public function mount()
    {
        $this->mood = session('chooseMood', 'happy');
    }


    #[On('follow-status-changed')]
    public function handleFollowChange()
    {
        // No logic needed, just triggering a re-render
    }

    public function render()
    {
        $search = $this->search;
        $filter = $this->filter;
        
        if ($filter === 'following') {
            $users = auth()->user()->followings();
            
            // Apply search to following users
            if ($search) {
                // Re-implementing search logic or reuse scope if possible. 
                // Relations return Relation instance, not Builder, but scopes usually work.
                // However, scopes are static-like on models or builders.
                // $users->searchUsers($search) won't work on BelongsToMany relation directly if it doesn't forward scopes effectively or if scope expects Builder.
                // It's safer to use the closure as in controller or try to call scope.
                 $users->where(function ($q) use ($search) {
                    $q->where('username', 'LIKE', "%{$search}%")
                      ->orWhereHas('userDetail', function ($q) use ($search) {
                          $q->where('fullname', 'LIKE', "%{$search}%");
                      });
                });
            }
            
            $users = $users->get();
        } else {
            $query = User::query()->where('role', '=', '0');
            $query->searchUsers($search);
            $users = $query->get();
        }

        return view('livewire.main.socials.index', [
            'users' => $users,
            'mood' => $this->mood
        ]);
    }
}
