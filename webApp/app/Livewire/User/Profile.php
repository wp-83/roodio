<?php
namespace App\Livewire\User;

use App\Models\Region;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;
use Livewire\Component;
use Livewire\Features\SupportFileUploads\WithFileUploads;

class Profile extends Component
{
    use WithFileUploads;
    public $photo;
    public $profilePhoto;
    public $fullname;
    public $email;
    public $dateOfBirth;
    public $regions;
    public $gender;
    public $countryId;
    public $username;

    public array $originalData = [];
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

    public function update()
    {
        $this->validate([
            'fullname'    => 'required|max:255',
            'email'       => 'required|email|max:255',
            'dateOfBirth' => 'required|date',
            'gender'      => 'required|in:1,0,null',
            'countryId'   => 'required|string|exists:regions,id',
        ]);

        auth()->user()->userDetail()->update([
            'fullname'    => $this->fullname,
            'email'       => $this->email,
            'dateOfBirth' => $this->dateOfBirth,
            'gender'      => $this->gender,
            'countryId'   => $this->countryId,
        ]);

        session()->flash('success', 'Profile updated successfully.');
    }

    public function mount()
    {
        $user               = Auth::user()->load('userDetail');
        $this->regions      = Region::all();
        $this->fullname     = $user->userDetail->fullname;
        $this->email        = $user->userDetail->email;
        $this->profilePhoto = $user->userDetail->profilePhoto;
        $this->dateOfBirth  = $user->userDetail->dateOfBirth;
        $this->gender       = $user->userDetail->gender;
        $this->countryId    = $user->userDetail->countryId;
        $this->username     = $user->username;
    }

    public function render()
    {
        return view('livewire.user.profile');
    }
}
