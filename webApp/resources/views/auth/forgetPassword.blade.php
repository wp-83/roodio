@extends('layouts.signIn')


@section('title', 'ROODIO - Change Password')


@push('script')
    <script type="text/javascript" src="{{ asset('js/auth/password.js') }}" defer></script>
@endpush


@section('headContent')
    <div class='flex flex-col items-center gap-1'>
        <img src="{{ asset('assets/logo/logo-with-text.png') }}" alt="logo" class='w-20 drop-shadow-sm drop-shadow-primary-50'>
        <p class='font-primary text-subtitle font-bold text-primary-85'>
            CHANGE PASSWORD
        </p>
        <p class='text-center text-primary-50 text-small mb-3'>
            Let's change your password so that you can login!
        </p>
    </div>
@endsection


@section('content')
    <form action="{{ route('auth.forgetPassword') }}" method='POST' id='changePassword'>
        @csrf {{-- cross site request forgery --}}
        <x-input type='password' id='password' icon='password' label='New Password' placeholder='Your new password...'>
            <x-slot:additionalContent>
                <button type='button' id='showPass' class='w-4 h-4 absolute z-4 right-2.5 bottom-2 flex items-center justify-center cursor-pointer md:bottom-2.5'>
                    <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed'>
                    <span class='absolute invisible' id='eye-open'>&#128065;</span>
                </button>
            </x-slot:additionalContent>
        </x-input>
        <x-input type='password' id='password_confirmation' icon='password' label='New Password Confirmation' placeholder='Input again your new password...'>
            <x-slot:additionalContent>
                <button type='button' id='showPassConfirm' class='w-4 h-4 absolute z-4 right-2.5 bottom-2 flex items-center justify-center cursor-pointer md:bottom-2.5'>
                    <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed-pass-confirm'>
                    <span class='absolute invisible' id='eye-open-pass-confirm'>&#128065;</span>
                </button>
            </x-slot:additionalContent>
        </x-input>
        <x-button behaviour='action' actionType='submit' form='changePassword' content='Change Password' class='min-w-full'></x-button>
    </form>
@endsection