@extends('layouts.master')


@section('title', 'ROODIO - SignUp')


@push('script')
    <script type="text/javascript" src="{{ asset('js/design/signUp-bg.js') }}" defer></script>
    <script type="text/javascript" src="{{ asset('js/design/particle-network.js') }}" defer></script>
@endpush


@section('bodyClass', 'relative')


@section('bodyContent')
    <div id="particle-canvas" class='relative min-h-lvh h-screen z-1'></div>
    <div class='absolute z-5 w-fit h-fit flex justify-center items-center left-1/2 top-1/2 -translate-1/2 @yield('marginStyle')'>
        <div class='border-primary-30 border-4 rounded-4xl bg-secondary-happy-10/85 w-sm h-max p-5 pt-5 font-secondaryAndButton shadow-xl shadow-primary-20/40 md:w-md @yield('customClass')'>
            <div class='flex flex-col items-center gap-1'>
                @yield('mainImage')
                <p class='font-primary text-subtitle font-bold text-primary-85 -mb-3'>
                    @yield('mainTitle')
                </p>
                <p class='text-center text-primary-50 text-small mb-2'>
                    @yield('description')
                </p>
            </div>
            <div>
                @yield('content')
            </div>
        </div>
    </div>
@endsection
