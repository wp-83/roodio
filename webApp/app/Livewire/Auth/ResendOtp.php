<?php
namespace App\Livewire\Auth;

use App\Http\Controllers\OtpController;
use Livewire\Component;

class ResendOtp extends Component
{
    public $fullname;
    public $email;
    public $gender;

    public $showTimer = true;

    public function mount()
    {
        $validated = session('register.step1');

        if ($validated) {
            $this->email    = $validated['email'];
            $this->fullname = $validated['fullname'];
            $this->gender   = $validated['gender'];
        }
    }

    public function sendOtp(OtpController $otpController)
    {
        if ($this->email) {
            $otpController->send($this->email, $this->fullname, $this->gender);
        }

        $this->dispatch('otp-resent');
    }

    public function render()
    {
        return view('livewire.auth.resendOtp');
    }
}
