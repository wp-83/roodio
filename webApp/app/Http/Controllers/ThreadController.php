<?php
namespace App\Http\Controllers;

use App\Models\Thread;
use App\Models\User;
use function Symfony\Component\Clock\now;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class ThreadController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index(Request $request)
    {
        $mood     = session('chooseMood', 'happy');
        $fullname = auth()->user()->userDetail->fullname;
        $user     = auth()->user();

        $search = $request->get('search');
        $filter = $request->get('filter', 'all');
        $query  = Thread::withCount('reactions')->orderByDesc('created_at');

        // Apply search if provided
        $query->search($search);

        if ($filter == 'created') {
            $query->where('userId', $user->id);
        } else if ($filter == 'following') {
            $followingIds = $user->followings()->pluck('users.id');
            $query->whereIn('userId', $followingIds);
        }

        $threads = $query->get();
        return view('main.threads.index', compact('threads', 'mood', 'fullname', 'user', 'search'));
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        return view('main.threads.create');
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $request->validate([
            'title'   => 'required|max:100',
            'content' => 'required|max:255',
        ]);

        $threadData             = $request->all();
        $threadData['userId']   = Auth::id();
        $threadData['datePost'] = now();

        Thread::create($threadData);
        return redirect()->route('threads.index')->with('success', 'Thread added successfully!');
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

    /**
     * Store a newly created resource in storage.
     */
    // public function reply(Request $request, Thread $thread)
    // {
    //     $request->validate([
    //         'content' => "required|min:2|max:255",
    //     ]);

    //     $reply             = $request->all();
    //     $reply['userId']   = Auth::id();
    //     $reply['threadId'] = $thread->id;

    //     Reply::create($reply);
    //     return redirect()->route('threads.index', $thread)->with('succes', 'berhasil membuat reply');
    // }

    /**
     * Store a newly created resource in storage.
     */
    // public function react(string $threadId)
    // {
    //     $userId = Auth::id();

    //     $thread = Thread::where('id', $threadId)->first();

    //     if (! $thread) {
    //         return response()->json([
    //             'message' => 'Thread not found',
    //         ], 404);
    //     }

    //     $reaction = Reactions::where('userId', $userId)
    //         ->where('threadId', $threadId)
    //         ->first();

    //     if ($reaction) {
    //         $reaction->delete();
    //     } else {
    //         Reactions::create([
    //             'userId'   => $userId,
    //             'threadId' => $threadId,
    //         ]);
    //     }

    //     $count = Reactions::where('threadId', $threadId)->count();

    //     return response()->json([
    //         'count' => $count,
    //     ]);
    // }
}
