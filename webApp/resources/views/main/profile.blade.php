@extends('layouts.master')


@section('title', 'ROODIO - My Profile')


@section('bodyClass', 'bg-primary-100 font-secondaryAndButton')


@section('bodyContent')
    @livewire('user.profile')
    {{-- <script>
        const setupSelect = (id) => {
            const el = document.querySelector(id);
            if (!el) return;

            const update = () => {
            if (el.value) {
                el.classList.add('bg-accent-20/60', 'text-shadedOfGray-100');
                el.classList.remove('text-shadedOfGray-60', 'bg-shadedOfGray-20/50', 'italic', 'bg-error-lighten/25');
            } else if(!el.classList.contains('error')) {
                el.classList.remove('bg-accent-20/60', 'text-shadedOfGray-100');
                el.classList.add('italic', 'bg-shadedOfGray-20/50', 'bg-error-lighten/25', 'border-error-dark');
            }
            };

            update();
            el.addEventListener('change', update);
        };

        setupSelect('#gender');
        setupSelect('#country');
    </script> --}}
@endsection
