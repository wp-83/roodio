@extends('layouts.master')


@section('title', 'ROODIO - My Profile')


@section('bodyClass', 'bg-primary-100 font-secondaryAndButton')


@section('bodyContent')
    @livewire('user.profile');
@endsection
