<?php
namespace App\Http\Controllers\SuperAdmin;

use App\Http\Controllers\Controller;
use App\Models\Region;
use App\Models\User;
use App\Models\userDetails;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Storage;
use Illuminate\Validation\Rule;
use Illuminate\Validation\Rules\Password;

class UserController extends Controller
{
    public function index(Request $request)
    {
        $query = User::query();

        if ($request->filled('role')) {
            $query->where('role', $request->get('role'));
        }

        if ($request->filled('search')) {
            $query->where('username', 'LIKE', '%' . $request->get('search') . '%');
        }

        $users        = $query->orderByDesc('created_at')->paginate(10);
        $totalUser    = User::get()->count();
        $totalNewUser = User::whereDate('created_at', today())->count();
        $regions      = Region::get();
        return view('superadmin.index', compact('users', 'totalUser', 'totalNewUser', 'regions'));
    }

    public function store(Request $request)
    {
        $path      = null;
        $validated = $request->validate([
            'username'     => 'required|max:25|unique:users,username',
            'password'     => [
                'required',
                'string',
                Password::min(8)->letters()->numbers(),
            ],
            'fullname'     => 'required|max:255',
            'email'        => 'required|email|max:255|unique:user_details,email',
            'dateOfBirth'  => 'required|date',
            'gender'       => 'nullable|in:0,1,-1',
            'countryId'    => 'required|string|exists:regions,id',
            'role'         => 'required|in:0,1,2',
            'profilePhoto' => 'image|max:5120',
        ]);

        if ($request->filled('profilePhoto')) {
            $path = Storage::disk('azure')->put(
                'images',
                $validated['profilePhoto']
            );
        }

        $user = User::create([
            'username' => $validated['username'],
            'password' => Hash::make($validated['password']),
            'role'     => $validated['role'],
        ]);

        if ($validated['gender'] == "-1") {
            $validated['gender'] = null;
        }

        userDetails::create([
            'userId'       => $user->id,
            'fullname'     => $validated['fullname'],
            'email'        => $validated['email'],
            'dateOfBirth'  => $validated['dateOfBirth'],
            'gender'       => $validated['gender'],
            'countryId'    => $validated['countryId'],
            'profilePhoto' => $path,
        ]);

        return back()->with(['success' => 'User successfully created!!']);
    }

    public function update(Request $request, User $user)
    {
        $validated = $request->validate([
            'username'     => ['required', 'max:25', Rule::unique('users')->ignore($user->id)],
            'password'     => ['nullable', 'string', Password::min(8)->letters()->numbers()],
            'fullname'     => 'required|max:255',
            'email'        => ['required', 'email', 'max:255', Rule::unique('user_details')->ignore($user->id, 'userId')],
            'dateOfBirth'  => 'required|date',
            'gender'       => 'nullable|in:0,1',
            'countryId'    => 'required|string|exists:regions,id',
            'role'         => 'required|in:0,1,2',
            'profilePhoto' => 'nullable|image|max:5120',
        ]);

        $userData = [
            'username' => $validated['username'],
            'role'     => $validated['role'],
        ];

        if ($request->filled('password')) {
            $userData['password'] = Hash::make($validated['password']);
        }

        $user->update($userData);

        $detailData = [
            'fullname'    => $validated['fullname'],
            'email'       => $validated['email'],
            'dateOfBirth' => $validated['dateOfBirth'],
            'gender'      => $validated['gender'],
            'countryId'   => $validated['countryId'],
        ];

        if ($request->hasFile('profilePhoto')) {
            if ($user->userDetail && $user->userDetail->profilePhoto) {
                Storage::disk('azure')->delete($user->userDetail->profilePhoto);
            }

            $path = $request->file('profilePhoto')->store('images', 'azure');

            $detailData['profilePhoto'] = $path;
        }

        $user->userDetail()->update($detailData);

        return back()->with(['success' => 'User successfully updated!!']);
    }

    public function destroy(User $user)
    {
        $user->delete();

        return back()->with(['success' => 'User successfully deleted!!']);
    }

    public function overview()
    {
        $totalUser     = User::count();
        $newUsersToday = User::whereDate('created_at', today())->count();
        $totalAdmins   = User::whereIn('role', [1, 2])->count();
        $totalMale     = User::whereHas('userDetail', function ($q) {
            $q->where('gender', 1);
        })->count();
        $totalFemale = User::whereHas('userDetail', function ($q) {
            $q->where('gender', 0);
        })->count();
        $users = User::with('userDetail')->orderByDesc('created_at')->limit(5)->get();

        $role0 = User::where('role', 0)->count();
        $role1 = User::where('role', 1)->count();
        $role2 = User::where('role', 2)->count();

        $total = $role0 + $role1 + $role2;

        $perc0 = $total > 0 ? round(($role0 / $total) * 100, 1) : 0;
        $perc1 = $total > 0 ? round(($role1 / $total) * 100, 1) : 0;
        $perc2 = $total > 0 ? round(($role2 / $total) * 100, 1) : 0;

        return view('superadmin.overview', compact('totalUser', 'newUsersToday', 'totalAdmins', 'totalMale', 'totalFemale', 'users',
            'role0', 'role1', 'role2',
            'perc0', 'perc1', 'perc2',
        ));
    }

    public function roles()
    {
        $role0 = User::where('role', 0)->count();
        $role1 = User::where('role', 1)->count();
        $role2 = User::where('role', 2)->count();
        return view('superadmin.roles', compact('role0', 'role1', 'role2', ));
    }
}
