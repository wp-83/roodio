@extends('layouts.signUp')


@section('title', 'ROODIO - Account')


@push('script')
    <script src="{{ asset('js/auth/password.js') }}" defer></script>
@endpush


@section('mainImage')
    <img src="{{ asset('assets/logo/logo-with-text.png') }}" alt="logo" class='w-20 drop-shadow-sm drop-shadow-primary-50'>
@endsection


@section('mainTitle', 'USER ACCOUNT')


@section('description', "Awesome! One step left with ROODIO!")


@section('content')
    <form action="{{ route('auth.account') }}" method="POST" id='account'>
        @csrf
        <x-input id='username' icon='user' label='Your Username' placeholder='Create your username...' value="{{ $username }}"  additionalInfo='You can customize your username.'></x-input>
        <x-input type='password' id='password' icon='password' label='Your Password' placeholder='Create your password...'>
            <x-slot:additionalContent>
                <button type='button' id='showPass' class='w-4 h-4 absolute z-4 right-2.5 bottom-2 flex items-center justify-center cursor-pointer md:bottom-2.5'>
                    <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed'>
                    <span class='absolute invisible' id='eye-open'>&#128065;</span>
                </button>
            </x-slot:additionalContent>
        </x-input>
        <x-input type='password' id='password_confirmation' icon='password' label='Password Confirmation' placeholder='Input again your password...'>
            <x-slot:additionalContent>
                <button type='button' id='showPassConfirm' class='w-4 h-4 absolute z-4 right-2.5 bottom-2 flex items-center justify-center cursor-pointer md:bottom-2.5'>
                    <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed-pass-confirm'>
                    <span class='absolute invisible' id='eye-open-pass-confirm'>&#128065;</span>
                </button>
            </x-slot:additionalContent>
        </x-input>
        <x-button behaviour='action' actionType='submit' form='account' content='Create Account' class='min-w-full'></x-button>
    </form>
@endsection
