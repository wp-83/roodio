@extends('layouts.signIn')


@section('title', 'ROODIO - Login')


@push('script')
    <script type="text/javascript" src="{{ asset('js/auth/password.js') }}" defer></script>
@endpush


@section('headContent')
    <div class='w-full flex flex-row h-max items-center justify-center font-primary text-title font-bold text-primary-70 md:text-hero animate-pulse'>
        <span>L</span>
        <div class='rounded-full w-12 h-12 bg-[conic-gradient(from_0deg,#7591DB_0%,#1F3A98_25%,#7591DB_50%,#1F3A98_75%,#7591DB_100%)] flex justify-center items-center mx-2 animate-spin md:w-16 md:h-16 border-2 border-primary-70'>
            <img src="{{ asset('assets/logo/logo-no-text.png') }}" alt="logo" class='w-6 bg-secondary-happy-60 rounded-full md:w-8'>
        </div>
        <span class='tracking-[.5rem]'>GI</span>
        <span>N</span>
    </div>
    <p class='mb-3 font-secondaryAndButton text-small text-primary-50 md:text-body-size md:tracking-[.05rem]'>Welcome back to ROODIO!</p>
@endsection


@section('content')
    @if (session('failed'))
        <div id="alert-2" class="flex sm:items-center p-4 mb-4 text-sm text-fg-danger-strong rounded-base bg-danger-soft" role="alert">
            <svg class="w-4 h-4 shrink-0 mt-0.5 md:mt-0" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 11h2v5m-2 0h4m-2.592-8.5h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/></svg>
            <span class="sr-only">Info</span>
            <div class="ms-2 text-sm ">
                <span class="font-medium me-1">{{ session('failed') }}</span>
            </div>
            <button type="button" class="ms-auto -mx-1.5 -my-1.5 bg-danger-soft text-fg-danger-strong rounded focus:ring-2 focus:ring-danger-medium p-1.5 hover:bg-danger-medium inline-flex items-center justify-center h-8 w-8 shrink-0 shrink-0" data-dismiss-target="#alert-2" aria-label="Close">
                <span class="sr-only">Close</span>
                <svg class="w-4 h-4" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18 17.94 6M18 18 6.06 6"/></svg>
            </button>
        </div>
    @endif
    <form action="{{ route('auth.login') }}" method='POST' id='login'>
        @csrf {{-- cross site request forgery --}}
        <x-input id='username' icon='user' label='Account' placeholder='Input your username or email...' value="{{ old('username') }}"></x-input>
        <x-input type='password' id='password' icon='password' label='Password' placeholder='Input your password...'>
            <x-slot:inlineContent>
                <x-button behaviour='navigation' navType='text' :navLink="route('user.verification')" content="Forget Password?" class='w-fit inline'></x-button>
            </x-slot:inlineContent>
            <x-slot:additionalContent>
                <button type='button' id='showPass' class='w-4 h-4 absolute z-4 right-2.5 bottom-2 flex items-center justify-center cursor-pointer md:bottom-2.5'>
                    <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed'>
                    <span class='absolute invisible' id='eye-open'>&#128065;</span>
                </button>
            </x-slot:additionalContent>
        </x-input>
        <div class='pt-2'>
            <x-button behaviour='action' actionType='submit' form='login' content='Login' class='min-w-full'></x-button>
            <div class='flex flex-row justify-center gap-1'>
                <p class='text-micro text-center md:text-small'>Don't have account?</p>
                <x-button behaviour='navigation' navType='text' :navLink="route('register')" content="Sign Up Here!"></x-button>
            </div>
        </div>
    </form>
@endsection
