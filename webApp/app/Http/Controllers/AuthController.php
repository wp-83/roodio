<?php
namespace App\Http\Controllers;

use App\Models\Region;
use App\Models\User;
use App\Models\userDetails;
use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Validation\Rules\Password;

class AuthController extends Controller
{
    public function loginView()
    {
        return view('auth.login');
    }

    public function login(Request $request)
    {
        $credentials = $request->validate([
            'username' => ['required'],
            'password' => ['required'],
        ]);

        if (Auth::attempt($credentials)) {
            $request->session()->regenerate();

            $user = Auth::user();

            if ($user->role === 0) {
                return redirect()->route('user.index');
            } elseif ($user->role === 1) {
                return redirect()->route('admin.overview');
            } elseif ($user->role === 2) {
                return redirect()->route('superadmin.users.overview');
            }
        }

        return back()->with('failed', 'username or password incorrect!');
    }

    public function emailVerificationView()
    {
        return view('auth.emailForgotPassword');
    }

    public function emailVerification(Request $request)
    {
        $validated = $request->validate([
            'email' => 'required|email|max:255|exists:user_details,email',
        ]);

        session()->put('forgot.step1', $validated);

        session(key: ['email_verification_passed' => true]);

        return redirect()->route('user.verification');
    }

    public function userVerificationView()
    {
        return view('auth.forgetPasswordValidation');
    }

    public function userVerification(Request $request, OtpController $otpController)
    {
        $otp = collect(range(1, 6))
            ->map(fn($i) => $request->input("otp-$i"))
            ->implode('');

        $request->merge(['otp' => $otp]);

        $request->validate([
            'otp' => 'required|digits:6',
        ], [
            'otp.required' => 'OTP code must be completed.',
            'otp.digits'   => 'OTP code must be 6 digits.',
        ]);

        $session = session('forgot.step1');
        $verify  = $otpController->verify($session['email'], $request['otp']);
        if ($verify) {
            session()->forget('email_verification_passed');
            session(['otp_forgot_passed' => true]);
            return redirect()->route('forgetPassword');
        } else {
            return back()->withErrors(['otp' => 'Invalid or expired OTP']);
        }
    }

    public function forgetPasswordView()
    {
        return view('auth.forgetPassword');
    }

    public function forgetPassword(Request $request)
    {
        $validated = $request->validate([
            'password'              => [
                'required',
                'string',
                'confirmed',
                Password::min(8)->letters()->numbers(),
            ],
            'password_confirmation' => 'required',
        ]);

        $session = session('forgot.step1');
        $user    = userDetails::where('email', $session['email'])->with('user')->first();
        $user->user->update([
            'password' => $validated['password'],
        ]);

        return redirect()->route('login');
    }

    public function registerView()
    {
        $regions = Region::orderBy('id')->get();
        return view('auth.register', compact('regions'));
    }

    public function register(Request $request)
    {
        $validated = $request->validate([
            'fullname' => 'required|max:255',
            'email'    => 'required|email|max:255|unique:user_details,email',
            'dob'      => 'required|date',
            'gender'   => 'required|in:1,0,null',
            'country'  => 'required|string|exists:regions,id',
        ]);

        session()->put('register.step1', $validated);

        session(['user_verification_passed' => true]);

        return redirect()->route('register.validation');
    }

    public function registerValidationView()
    {
        if (! session()->has('register.step1')) {
            return redirect()->route('register');
        }
        return view('auth.registerValidation');
    }

    public function registerValidation(Request $request, OtpController $otpController)
    {
        $otp = collect(range(1, 6))
            ->map(fn($i) => $request->input("otp-$i"))
            ->implode('');

        $request->merge(['otp' => $otp]);

        $request->validate([
            'otp' => 'required|digits:6',
        ], [
            'otp.required' => 'OTP code must be completed.',
            'otp.digits'   => 'OTP code must be 6 digits.',
        ]);

        $session = session('register.step1');
        $verify  = $otpController->verify($session['email'], $request['otp']);
        if ($verify) {
            session(['otp_passed' => true]);
            return redirect()->route('account');
        } else {
            return back()->withErrors(['otp' => 'Invalid or expired OTP']);
        }
    }

    public function accountView()
    {
        $datas     = session('register.step1');
        $firstName = trim(explode(' ', $datas['fullname'])[0]);
        $random    = random_int(100000, 999999);
        $symbols   = ['-', '#', '_', '.', '~'];
        $symbol    = collect($symbols)->random();

        $username = $firstName . $random . $symbol;
        return view('auth.account', compact('username'));
    }

    public function account(Request $request)
    {
        $validated = $request->validate([
            'username'              => 'required|max:25|unique:users,username',
            'password'              => [
                'required',
                'string',
                'confirmed',
                Password::min(8)->letters()->numbers(),
            ],
            'password_confirmation' => 'required',
        ]);

        $user = User::create($validated);

        $datas                = session('register.step1');
        $datas['dateOfBirth'] = $datas['dob'];
        $datas['countryId']   = $datas['country'];
        $datas['dateOfBirth'] = Carbon::createFromFormat(
            'm/d/Y',
            $datas['dateOfBirth']
        )->format('Y-m-d');
        unset($datas['dob']);
        unset($datas['country']);
        $datas['userId'] = $user->id;

        userDetails::create($datas);

        return redirect()->route('login');
    }

    public function logout(Request $request)
    {
        Auth::logout();

        $request->session()->invalidate();

        $request->session()->regenerateToken();

        return redirect()->route('roodio');
    }
}
