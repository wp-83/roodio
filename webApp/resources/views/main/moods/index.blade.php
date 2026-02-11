@extends('layouts.main')

@php
    // dd($weekly, $monthly, $yearly);
    dd($weekly[0]->mood->type);
@endphp

@section('title', 'ROODIO - Moods')
