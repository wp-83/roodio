<?php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\Playlists;
use Illuminate\Http\Request;

class SongController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        $playlists    = Playlists::orderByDesc('created_at')->get();
        $username     = auth()->user()->username;
        $fullname     = auth()->user()->userDetail->fullname;
        $profilePhoto = auth()->user()->userDetail->profilePhoo;
        return view('main.index', compact('playlists', 'username', 'fullname', 'profilePhoto'));
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        //
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        //
    }
}
