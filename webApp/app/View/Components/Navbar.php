<?php
namespace App\View\Components;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class Navbar extends Component
{
    public $username;
    public $fullname;
    public $profilePhoto;
    public $mood;
    public $showSearch;
    /**
     * Create a new component instance.
     */
    public function __construct()
    {
        $this->username     = auth()->user()->username;
        $this->fullname     = auth()->user()->userDetail->fullname;
        $this->profilePhoto = auth()->user()->userDetail->profilePhoto;
        $this->mood         = session('chooseMood', 'happy');
        $this->showSearch   = request()->routeIs('user.playlists', 'threads.index', 'socials.index');
    }

    /**
     * Get the view / contents that represent the component.
     */
    public function render(): View | Closure | string
    {
        return view('components.navbar');
    }
}
