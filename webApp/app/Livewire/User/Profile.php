<?php
namespace App\Livewire\User;

use App\Models\Region;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;
use Illuminate\Validation\Rules\Password;
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
    public $newUsername;
    public $confirmPassword;
    public $currentPassword;
    public $newPassword;
    public $newPasswordConfirmation;
    public $deleteConfirmationPassword;
    public $passwordLastChanged;

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
        $this->profilePhoto = $path;

        $this->reset('photo');

        $this->dispatch(event: 'photo-updated');

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

        $gender = $this->gender === 'null' ? null : $this->gender;

        auth()->user()->userDetail()->update([
            'fullname'    => $this->fullname,
            'email'       => $this->email,
            'dateOfBirth' => $this->dateOfBirth,
            'gender'      => $gender,
            'countryId'   => $this->countryId,
        ]);

        session()->flash('success', 'Profile updated successfully.');
    }

    public function updateUsername()
    {
        $this->validate([
            'newUsername'     => 'required|max:25',
            'confirmPassword' => 'required|current_password',
        ]);

        auth()->user()->update([
            'username' => $this->newUsername,
        ]);

        $this->username = $this->newUsername;

        $this->reset(['newUsername', 'confirmPassword']);

        $this->dispatch('username-updated');

        $this->dispatch('username-updated');

        $this->passwordLastChanged = now(); // Username change updates updated_at
        session()->flash('success', 'Username updated successfully.');
    }

    public function updatePassword()
    {
        $this->validate([
            'currentPassword'         => 'required|current_password',
            'newPassword'             => ['required',
                'string',
                'same:newPasswordConfirmation',
                Password::min(8)->letters()->numbers()],
            'newPasswordConfirmation' => 'required',
        ]);

        auth()->user()->update([
            'password' => $this->newPassword,
        ]);

        $this->currentPassword = $this->newPassword;

        $this->reset(['currentPassword', 'newPassword', 'newPasswordConfirmation']);

        $this->dispatch('password-updated');

        $this->dispatch('password-updated');

        $this->passwordLastChanged = now();
        session()->flash('success', 'Password updated successfully.');
    }

    public function deleteAccount()
    {
        $this->validate([
            'deleteConfirmationPassword' => 'required|current_password',
        ]);

        auth()->user()->delete();

        return redirect()->route('auth.login');
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
        $this->passwordLastChanged = $user->updated_at;
    }

    public function render()
    {
        return view('livewire.user.profile');
    }
}
