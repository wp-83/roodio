<?php
namespace App\Livewire\User;

use App\Models\Region;
use App\Services\AzureSasService;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;
use Livewire\Component;
use Livewire\Features\SupportFileUploads\WithFileUploads;

class Profile extends Component
{
    use WithFileUploads;
    public $photo;

    protected AzureSasService $azure;

    public function uploadPhoto()
    {
        $this->validate([
            'photo' => 'image|max:5120',
        ]);

        $user = auth()->user();

        $path = Storage::disk('azure')->put(
            'images',
            $this->photo
        );

        $user->userDetail()->update(['profilePhoto' => $path]);

        session()->flash('success', 'Image successfully uploaded.');
    }

    public function mount()
    {
        $this->azure    = app(AzureSasService::class);
        $this->photoUrl = $this->azure->blob('image.jpg', 30);
    }

    public function render()
    {
        $user    = Auth::user()->load('userDetail');
        $regions = Region::all();

        $azure = app(AzureSasService::class);

        $photoUrl = $azure->blob($user->userDetail()->profilePhoto, 30);
        dd($photoUrl);
        return view('livewire.user.profile', compact('user', 'regions', 'photoUrl'));
    }
}
