<?php
namespace App\Livewire\User;

use App\Models\Reply;
use App\Models\Thread;
use Illuminate\Support\Facades\Auth;
use Livewire\Component;

class ThreadBox extends Component
{
    public Thread $thread;
    public $mainUser;
    public $content = '';
    public $mood;
    public $isReplyable;

    public function mount($thread, $mainUser, $mood, $isReplyable = true)
    {
        $this->thread      = $thread;
        $this->mainUser    = $mainUser;
        $this->mood        = $mood;
        $this->isReplyable = $isReplyable;
    }

    public function postReply()
    {
        $this->validate([
            'content' => 'required|string|max:1000',
        ]);

        Reply::create([
            'userId'   => Auth::id(),
            'threadId' => $this->thread->id,
            'content'  => $this->content,
            'datePost' => now(),
        ]);

        // Reset input
        $this->content = '';

        // Refresh data thread agar reply baru muncul di list
        $this->thread->refresh();

        // Kirim sinyal ke JS untuk scroll ke bawah
        $this->dispatch('reply-posted', threadId: $this->thread->id);
    }

    public function render()
    {
        // Styling Configuration
        $styles = [
            'bgContainer'         => [
                'happy'   => 'bg-secondary-happy-10/95',
                'sad'     => 'bg-secondary-sad-10/95',
                'relaxed' => 'bg-secondary-relaxed-10/95',
                'angry'   => 'bg-secondary-angry-10/95',
            ],
            'elementColor'        => [
                'happy'   => '#FF8E2B',
                'sad'     => '#6A4FBF',
                'relaxed' => '#28C76F',
                'angry'   => '#E63946',
            ],
            'borderMood'          => [
                'happy'   => 'border-secondary-happy-100',
                'sad'     => 'border-secondary-sad-100',
                'relaxed' => 'border-secondary-relaxed-100',
                'angry'   => 'border-secondary-angry-100',
            ],
            'textMood'            => [
                'happy'   => 'text-secondary-happy-100',
                'sad'     => 'text-secondary-sad-100',
                'relaxed' => 'text-secondary-relaxed-100',
                'angry'   => 'text-secondary-angry-100',
            ],
            'bgMoodStyle'         => [
                'happy'   => 'bg-secondary-happy-30',
                'sad'     => 'bg-secondary-sad-30',
                'relaxed' => 'bg-secondary-relaxed-30',
                'angry'   => 'bg-secondary-angry-30',
            ],
            'scrollbarTheme'      => [
                'happy'   => 'scrollbar-thumb-secondary-happy-85/75 scrollbar-track-transparent',
                'sad'     => 'scrollbar-thumb-secondary-sad-85/75 scrollbar-track-transparent',
                'relaxed' => 'scrollbar-thumb-secondary-relaxed-85/75 scrollbar-track-transparent',
                'angry'   => 'scrollbar-thumb-secondary-angry-85/75 scrollbar-track-transparent',
            ],
            'borderTextareaStyle' => [
                'happy'   => 'border-secondary-happy-100',
                'sad'     => 'border-secondary-sad-100',
                'angry'   => 'border-secondary-angry-100',
                'relaxed' => 'border-secondary-relaxed-100',
            ],
        ];

        return view('livewire.user.thread-box', ['styles' => $styles]);
    }
}
