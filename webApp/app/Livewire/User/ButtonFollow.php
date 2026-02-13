<?php
namespace App\Livewire\User;

use App\Models\Follow;
use Illuminate\Support\Facades\Auth;
use Livewire\Attributes\On;
use Livewire\Component;

class ButtonFollow extends Component
{
    public $userId;

    public $isFollowing = false;
    public $mood        = 'happy';
    public $customClass = '';
    public $customStyle = '';

    public function mount($userId, $mood = 'happy', $customClass = '', $customStyle = '')
    {
        $this->userId      = $userId;
        $this->mood        = $mood;
        $this->customClass = $customClass;
        $this->customStyle = $customStyle;

        if (Auth::check()) {
            $this->isFollowing = Follow::where('userId', Auth::id())
                ->where('followedId', $this->userId)
                ->exists();
        }
    }

    public function toggle()
    {
        if (! Auth::check()) {
            return redirect()->route('login');
        }

        $currentUserId = Auth::id();

        $following = Follow::where('userId', $currentUserId)
            ->where('followedId', $this->userId)
            ->first();

        if ($following) {
            $following->delete();
            $status = false;
        } else {
            Follow::create([
                'userId'     => $currentUserId,
                'followedId' => $this->userId,
            ]);
            $status = true;
        }

        $this->isFollowing = $status;

        $this->dispatch('follow-status-changed', followedId: $this->userId, status: $status);
    }

    #[On('follow-status-changed')]
    public function updateFollowStatus($followedId, $status)
    {
        if ($this->userId == $followedId) {
            $this->isFollowing = $status;
        }
    }

    public function render()
    {
        return view('livewire.button-follow');
    }
}
