@props([
    'route',
    'codeAmount' => 6
])


@php
    $idx = 1;
    $amount = $codeAmount;
@endphp


@extends('layouts.signUp')


@section('title', 'ROODIO - OTP Authentication')


@push('script')
    <script src="{{ asset('js/auth/otp.js') }}" defer></script>
@endpush


@section('mainImage')
    <img src="{{ asset('assets/icons/otp.svg') }}" alt="logo" class='w-20'>
@endsection


@section('mainTitle', 'OTP VERIFICATION')


@section('description')
    <span>We have sent you the 6-digit code <b>to your email for Two-Factor Authentication (2FA). Please enter the code to continue the process.</b></span>
@endsection

@section('bodyClass', 'overflow-y-hidden')

@section('content')
    <form action="{{ route($route) }}" method="POST" id='otp-form'>
        @csrf
        <div class='flex flex-col mt-4 mb-8 otp-container'>
            <div class='flex flex-row justify-center gap-5 items-center'>
                @while ($idx <= $amount)
                    <input type="text" maxlength="1" inputmode="numeric" name="otp-{{ $idx }}" id="otp-{{ $idx }}" autocomplete="off" placeholder="*" class='not-placeholder-shown:bg-white not-placeholder-shown:border-primary-30 font-bold text-center text-paragraph border-2 rounded-md px-1.5 py-0.5 w-10 h-12 placeholder:text-paragraph focus:border-secondary-happy-100 focus:bg-secondary-happy-20/65 hover:bg-shadedOfGray-30/45 ease-in-out duration-125 {{ $errors->has('otp') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'bg-shadedOfGray-20/50 border-shadedOfGray-50' }}'>
                    @php ($idx++)
                @endwhile
            </div>
            @if ($errors->has('otp'))
                <div class="error-message text-center mt-1 mb-2">
                    {{ $errors->first('otp') }}
                </div>
            @endif
        </div>
        <x-button behaviour='action' actionType='submit' form='otp-form' content='Verify Account' class='min-w-full'></x-button>
        <div class="mt-4">
            @livewire('auth.resendOtp')
        </div>
    </form>
@endsection
