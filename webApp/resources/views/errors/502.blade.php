@extends('layouts.error', [
    'errorCode' => '502',
    'errorIdentity' => 'Bad Gateway',
    'errorDescription' => 'Something went wrong while connecting to the server. Please try again later.'
])