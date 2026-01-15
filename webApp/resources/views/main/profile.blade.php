@extends('layouts.master')

@section('title', 'My Profile')

@section('bodyClass', 'bg-shadedOfGray-10 min-h-screen py-10 px-4 font-secondaryAndButton text-shadedOfGray-85')

@section('bodyContent')
    @livewire('user.profile');
@endsection
