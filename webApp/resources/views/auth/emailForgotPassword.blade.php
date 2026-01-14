@extends('layouts.signIn')


@section('title', 'ROODIO - Email Verification')


@push('script')
    <script type="text/javascript" src="{{ asset('js/auth/password.js') }}" defer></script>
@endpush


@section('headContent')
    <div class='flex flex-col items-center gap-1'>
        <img src="{{ asset('assets/icons/email.svg') }}" alt="logo" class='w-20'>
        <p class='font-primary text-subtitle font-bold text-primary-85'>
            VERIFY ACCOUNT
        </p>
        <p class='text-center text-primary-50 text-small mb-7'>
            Let's verify your account before changing the password!
        </p>
    </div>
@endsection


@section('content')
    <form action="{{ route('auth.forgetPassword') }}" method='POST' id='changePassword'>
        @csrf {{-- cross site request forgery --}}
        <x-input type='password' id='password' icon='email' label='Your Email' placeholder='Your email account...'></x-input>
        <x-button behaviour='action' actionType='submit' form='changePassword' content='Verify Email' class='min-w-full'></x-button>
    </form>
@endsection