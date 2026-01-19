<?php
namespace App\Livewire\User;

use App\Models\Follow;
use Livewire\Attributes\On;
use Livewire\Component;

class ButtonFollow extends Component
{
    public $isFollowing = false;
    public $mood        = 'happy';
    public $thread;
    public function mount($thread)
    {
        $this->thread      = $thread;
        $this->mood        = session('chooseMood');
        $this->isFollowing = Follow::where('followedId', $thread->user->id)->exists();
    }

    public function toggle()
    {
        $following = Follow::where('followedId', $this->thread->user->id)->first();
        if ($following) {
            $following->delete();
            $status = false;
        } else {
            Follow::create([
                'userId'     => auth()->user()->id,
                'followedId' => $this->thread->user->id,
            ]);
            $status = true;
        }

        $this->isFollowing = $status;
        $this->dispatch('follow-status-changed', followedId: $this->thread->user->id, status: $status);
    }

    #[On('follow-status-changed')]
    public function updateFollowStatus($followedId, $status)
    {
        if ($this->thread->user->id == $followedId) {
            $this->isFollowing = $status;
        }
    }

    public function render()
    {
        return view('livewire.button-follow');
    }
}
