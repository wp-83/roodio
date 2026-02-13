<?php
namespace App\Livewire\Auth;

use App\Http\Controllers\OtpController;
use Illuminate\Support\Facades\RateLimiter;
use Livewire\Component;

class ResendOtp extends Component
{
    public $fullname;
    public $email;
    public $gender;
    public $secondsRemaining = 60;

    public function mount(OtpController $otpController)
    {
        if (session('register.step1')) {
            $validated      = session('register.step1');
            $this->email    = $validated['email'];
            $this->fullname = $validated['fullname'];
            $this->gender   = $validated['gender'] == "null" ? 9 : $validated['gender'];
        } else if (session('forgot.step1')) {
            $validated      = session('forgot.step1');
            $this->email    = $validated['email'];
            $this->fullname = '';
            $this->gender   = 2;
        }

        $key = 'otp-send:' . $this->email . ':' . request()->ip();

        if (RateLimiter::tooManyAttempts($key, 1)) {
            $this->secondsRemaining = RateLimiter::availableIn($key);
        } else {
            if ($this->email) {
                $otpController->send($this->email, $this->fullname, $this->gender);
                RateLimiter::hit($key, 60);
                $this->secondsRemaining = 60;
            }
        }
    }

    public function sendOtp(OtpController $otpController)
    {
        $key = 'otp-send:' . $this->email . ':' . request()->ip();

        if (RateLimiter::tooManyAttempts($key, 1)) {
            return;
        }

        if ($this->email) {
            $otpController->send($this->email, $this->fullname, $this->gender);
            RateLimiter::hit($key, 60);
            $this->secondsRemaining = 60;
        }

        $this->dispatch('otp-resent');
    }

    public function render()
    {
        return view('livewire.auth.resendOtp');
    }
}
