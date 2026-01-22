@extends('layouts.superadmin.master')

@section('title', 'User Management')
@section('page_title', 'User Management')
@section('page_subtitle', 'Manage system users permissions')

@section('content')

    {{-- 1. FLASH MESSAGE (SUCCESS) --}}
    @if(session('success'))
        <div id="flashMessage" class="mb-6 bg-secondary-relaxed-20 border border-secondary-relaxed-100 text-secondary-relaxed-100 px-4 py-3 rounded-xl relative shadow-sm flex items-center gap-3">
            <i class="fa-solid fa-circle-check text-xl"></i>
            <span class="block sm:inline font-medium">{{ session('success') }}</span>
            <button onclick="document.getElementById('flashMessage').remove()" class="absolute top-0 bottom-0 right-0 px-4 py-3">
                <i class="fa-solid fa-xmark"></i>
            </button>
        </div>
    @endif

    {{-- 2. STATS CARDS --}}
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-6 mb-8">
        {{-- Card 1: Total Users --}}
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center justify-between">
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1">Total Users</p>
                <h3 class="font-primary text-2xl lg:text-3xl text-primary-100 font-bold">{{ $totalUser }}</h3>
            </div>
            <div class="w-12 h-12 lg:w-14 lg:h-14 rounded-xl bg-secondary-happy-20 flex items-center justify-center text-secondary-happy-100 text-xl lg:text-2xl">
                <i class="fa-solid fa-users"></i>
            </div>
        </div>

        {{-- Card 2: New Registrations --}}
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center justify-between">
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1">New Registrations</p>
                <h3 class="font-primary text-2xl lg:text-3xl text-primary-100 font-bold">{{ $totalNewUser }}</h3>
            </div>
            <div class="w-12 h-12 lg:w-14 lg:h-14 rounded-xl bg-accent-20 flex items-center justify-center text-accent-100 text-xl lg:text-2xl">
                <i class="fa-solid fa-user-plus"></i>
            </div>
        </div>
    </div>

    {{-- 3. TABLE CONTAINER --}}
    <div class="bg-white rounded-2xl shadow-lg border border-shadedOfGray-10 overflow-visible">

        {{-- TOOLBAR --}}
        <div class="p-4 lg:p-6 border-b border-shadedOfGray-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
            {{-- SEARCH FORM --}}
            <form action="{{ route('superadmin.users.index') }}" method="GET" class="w-full md:w-80 lg:w-96">
                @if(request()->has('role'))
                    <input type="hidden" name="role" value="{{ request('role') }}">
                @endif
                <div class="relative w-full">
                    <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40 pointer-events-none">
                        <i class="fa-solid fa-magnifying-glass"></i>
                    </span>
                    <input type="text" name="search" value="{{ request('search') }}" placeholder="Search by username..."
                        class="w-full pl-10 pr-10 py-2.5 lg:py-3 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm transition-all">

                    @if(request()->filled('search'))
                        <a href="{{ route('superadmin.users.index', request()->except(['search', 'page'])) }}" class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 hover:text-accent-100 cursor-pointer transition-colors" title="Clear Search">
                            <i class="fa-solid fa-xmark"></i>
                        </a>
                    @endif
                    <button type="submit" class="hidden"></button>
                </div>
            </form>

            <div class="flex gap-2 lg:gap-3 w-full md:w-auto">
                {{-- FILTER BUTTON & POPUP --}}
                <div class="relative flex-1 md:flex-none">
                    <button onclick="toggleFilter()" id="filterBtn" class="w-full px-4 lg:px-5 py-2.5 lg:py-3 rounded-xl border border-shadedOfGray-20 text-primary-70 font-medium hover:bg-shadedOfGray-10 transition-colors flex items-center justify-center gap-2 text-sm">
                        <i class="fa-solid fa-filter"></i>
                        <span class="hidden sm:inline">Filter</span>
                        @if(request()->has('role'))
                            <span class="w-2 h-2 rounded-full bg-accent-100"></span>
                        @endif
                    </button>
                    <div id="filterPopup" class="hidden absolute top-full right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-shadedOfGray-10 z-50 overflow-hidden">
                        <div class="px-4 py-3 border-b border-shadedOfGray-10 bg-primary-10/10">
                            <p class="text-xs font-bold text-shadedOfGray-60 uppercase tracking-wider">Filter by Role</p>
                        </div>
                        <div class="py-2 flex flex-col">
                            <a href="{{ route('superadmin.users.index', ['role' => 0]) }}" class="px-4 py-2.5 text-sm hover:bg-primary-10/30 transition-colors flex items-center justify-between {{ request('role') === '0' ? 'text-accent-100 font-bold bg-accent-20' : 'text-primary-100' }}">
                                <span>User</span> @if(request('role') === '0') <i class="fa-solid fa-check"></i> @endif
                            </a>
                            <a href="{{ route('superadmin.users.index', ['role' => 1]) }}" class="px-4 py-2.5 text-sm hover:bg-primary-10/30 transition-colors flex items-center justify-between {{ request('role') === '1' ? 'text-accent-100 font-bold bg-accent-20' : 'text-primary-100' }}">
                                <span>Admin</span> @if(request('role') === '1') <i class="fa-solid fa-check"></i> @endif
                            </a>
                            <a href="{{ route('superadmin.users.index', ['role' => 2]) }}" class="px-4 py-2.5 text-sm hover:bg-primary-10/30 transition-colors flex items-center justify-between {{ request('role') === '2' ? 'text-accent-100 font-bold bg-accent-20' : 'text-primary-100' }}">
                                <span>Superadmin</span> @if(request('role') === '2') <i class="fa-solid fa-check"></i> @endif
                            </a>
                        </div>
                        @if(request()->has('role'))
                            <div class="border-t border-shadedOfGray-10">
                                <a href="{{ route('superadmin.users.index') }}" class="block px-4 py-3 text-xs text-center text-shadedOfGray-60 hover:text-accent-100 font-bold transition-colors">Reset Filter</a>
                            </div>
                        @endif
                    </div>
                </div>

                <button onclick="toggleCreateModal()" class="flex-1 md:flex-none bg-accent-100 hover:bg-accent-85 text-white px-4 lg:px-6 py-2.5 lg:py-3 rounded-xl font-medium shadow-lg shadow-accent-50/50 transition-all flex items-center justify-center gap-2 text-sm">
                    <i class="fa-solid fa-plus"></i> <span class="whitespace-nowrap">Add User</span>
                </button>
            </div>
        </div>

        {{-- TABLE WRAPPER --}}
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse min-w-[800px]">
                <thead class="bg-primary-100 text-white">
                    <tr>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide">User</th>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide">Role</th>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide">Joined Date</th>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide text-right">Actions</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-shadedOfGray-10">
                    @foreach($users as $user)
                        <tr class="hover:bg-primary-10/30 transition-colors duration-150 group">
                            <td class="px-6 py-4">
                                <div class="flex items-center gap-3">
                                    @if ($user->userDetail?->profilePhoto)
                                        <img class="w-10 h-10 rounded-full object-cover" src="{{ config('filesystems.disks.azure.url') . '/' . $user->userDetail?->profilePhoto }}" alt="{{ $user->userDetail?->fullname }}">
                                    @else
                                        <p class='w-10 h-10 rounded-full object-cover text-title font-primary font-bold flex items-center justify-center bg-shadedOfGray-10 text-primary-100'>
                                            {{ Str::length($user->username) > 0 ? Str::upper(Str::substr($user->username, 0, 1)) : '?' }}
                                        </p>
                                    @endif
                                    <div>
                                        <p class="text-sm lg:text-base font-bold text-primary-100 group-hover:text-accent-100 transition-colors">{{ $user->userDetail?->fullname ?? 'No Name' }}</p>
                                        <p class="text-xs text-shadedOfGray-60"><span>@</span>{{ $user->username }}</p>
                                    </div>
                                </div>
                            </td>
                            <td class="px-6 py-4">
                                @php
                                    $roleName = $user->role == 0 ? 'User' : ($user->role == 1 ? 'Admin' : 'Superadmin');
                                    $roleClass = $user->role == 0 ? 'bg-primary-20 text-primary-100' : ($user->role == 1 ? 'bg-secondary-happy-20 text-secondary-happy-100' : 'bg-accent-20 text-accent-100');
                                @endphp
                                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold {{ $roleClass }}">{{ $roleName }}</span>
                            </td>
                            <td class="px-6 py-4 text-sm text-shadedOfGray-70">{{ $user->created_at->format('d M Y') }}</td>
                            <td class="px-6 py-4 text-right">
                                <div class="flex items-center justify-end gap-2">
                                    <button onclick="openEditModal({{ json_encode($user) }}, '{{ $user->userDetail?->profilePhoto }}')"
                                        class="w-8 h-8 rounded-lg flex items-center justify-center text-accent-100 hover:bg-accent-20 transition-colors">
                                        <i class="fa-solid fa-pen-to-square"></i>
                                    </button>
                                    <button onclick="openDeleteModal('{{ $user->id }}', '{{ $user->username }}')"
                                        class="w-8 h-8 rounded-lg flex items-center justify-center text-secondary-angry-100 hover:bg-secondary-angry-20 transition-colors"
                                        title="Delete User">
                                        <i class="fa-solid fa-trash-can"></i>
                                    </button>
                                </div>
                            </td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>

        {{-- PAGINATION --}}
        <div class="p-4 lg:p-6 border-t border-shadedOfGray-10 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p class="text-sm text-shadedOfGray-60 text-center sm:text-left">
                Showing <span class="font-bold text-primary-100">{{ $users->firstItem() ?? 0 }}</span> to <span class="font-bold text-primary-100">{{ $users->lastItem() ?? 0 }}</span> of <span class="font-bold text-primary-100">{{ $users->total() }}</span> users
            </p>
            <div>{{ $users->appends(request()->query())->links('pagination.superadmin') }}</div>
        </div>
    </div>

    {{-- 4. CREATE USER MODAL --}}
    <div id="createUserModal" class="hidden fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
        <div class="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            <div class="fixed inset-0 bg-primary-100/75 transition-opacity" aria-hidden="true" onclick="toggleCreateModal()"></div>
            <span class="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

            <div class="relative inline-block align-bottom bg-white rounded-2xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-3xl w-full">
                {{-- Header --}}
                <div class="bg-primary-100 px-6 py-4 flex justify-between items-center">
                    <h3 class="text-lg font-bold text-white font-primary tracking-wide">Create New User</h3>
                    <button onclick="toggleCreateModal()" class="text-primary-30 hover:text-white transition-colors"><i class="fa-solid fa-xmark text-xl"></i></button>
                </div>

                {{-- Form dengan ID baru: createUserForm --}}
                <form id="createUserForm" action="{{ route('superadmin.users.store') }}" method="POST" enctype="multipart/form-data" class="p-6 lg:p-8">
                    @csrf

                    {{-- Global Errors --}}
                    @if ($errors->any())
                        <div id="globalErrorBox" class="mb-6 p-4 rounded-xl bg-red-50 border border-red-200">
                            <p class="text-red-600 font-bold text-sm mb-1">Please fix the following errors:</p>
                            <ul class="list-disc list-inside text-xs text-red-500">
                                @foreach ($errors->all() as $error)
                                    <li>{{ $error }}</li>
                                @endforeach
                            </ul>
                        </div>
                    @endif

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">

                        {{-- SECTION: ACCOUNT INFO --}}
                        <div class="md:col-span-2">
                            <p class="text-xs font-bold text-accent-100 uppercase tracking-wider mb-1">Account Information</p>
                            <div class="h-0.5 w-full bg-shadedOfGray-10 mb-4"></div>
                        </div>

                        {{-- Username --}}
                        <div class="space-y-1">
                            <label for="username" class="text-sm font-bold text-primary-100">Username <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-at"></i></span>
                                <input type="text" name="username" id="username" maxlength="25" placeholder="e.g. johndoe" value="{{ old('username') }}"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border {{ $errors->has('username') ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                            @error('username') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                        {{-- Email --}}
                        <div class="space-y-1">
                            <label for="email" class="text-sm font-bold text-primary-100">Email Address <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-regular fa-envelope"></i></span>
                                <input type="email" name="email" id="email" maxlength="255" placeholder="john@example.com" value="{{ old('email') }}"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border {{ $errors->has('email') ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                            @error('email') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                        {{-- Password --}}
                        <div class="space-y-1">
                            <label for="password" class="text-sm font-bold text-primary-100">Password <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-lock"></i></span>
                                <input type="password" name="password" id="password" placeholder="••••••••"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border {{ $errors->has('password') ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                            @error('password') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                        {{-- Role --}}
                        <div class="space-y-1">
                            <label for="role" class="text-sm font-bold text-primary-100">Role <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-shield-halved"></i></span>
                                <select name="role" id="role" class="w-full pl-10 pr-4 py-2.5 rounded-xl border {{ $errors->has('role') ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    <option value="0" {{ old('role') == '0' ? 'selected' : '' }}>User (Listener)</option>
                                    <option value="1" {{ old('role') == '1' ? 'selected' : '' }}>Admin</option>
                                    <option value="2" {{ old('role') == '2' ? 'selected' : '' }}>Superadmin</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @error('role') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                        {{-- SECTION: PERSONAL INFO --}}
                        <div class="md:col-span-2 mt-2">
                            <p class="text-xs font-bold text-accent-100 uppercase tracking-wider mb-1">Personal Details</p>
                            <div class="h-0.5 w-full bg-shadedOfGray-10 mb-4"></div>
                        </div>

                        {{-- Fullname --}}
                        <div class="space-y-1 md:col-span-2">
                            <label for="fullname" class="text-sm font-bold text-primary-100">Full Name <span class="text-red-500">*</span></label>
                            <input type="text" name="fullname" id="fullname" maxlength="255" placeholder="e.g. Johnathan Doe" value="{{ old('fullname') }}"
                                class="w-full px-4 py-2.5 rounded-xl border {{ $errors->has('fullname') ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                             @error('fullname') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                        {{-- Date of Birth --}}
                        <div class="space-y-1">
                            <label for="dateOfBirth" class="text-sm font-bold text-primary-100">Date of Birth <span class="text-red-500">*</span></label>
                            <input type="date" name="dateOfBirth" id="dateOfBirth" value="{{ old('dateOfBirth') }}"
                                class="w-full px-4 py-2.5 rounded-xl border {{ $errors->has('dateOfBirth') ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            @error('dateOfBirth') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                        {{-- Gender --}}
                        <div class="space-y-1">
                            <label for="gender" class="text-sm font-bold text-primary-100">Gender <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <select name="gender" id="gender" class="w-full px-4 py-2.5 rounded-xl border {{ $errors->has('gender') ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    <option value="" disabled {{ old('gender') === null ? 'selected' : '' }}>Select Gender</option>
                                    <option value="1" {{ old('gender') == '1' ? 'selected' : '' }}>Male</option>
                                    <option value="0" {{ old('gender') == '0' ? 'selected' : '' }}>Female</option>
                                    <option value="" {{ old('gender') !== null && old('gender') == '' ? 'selected' : '' }}>Prefer not to say</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @error('gender') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                        {{-- Country --}}
                        <div class="space-y-1">
                            <label for="countryId" class="text-sm font-bold text-primary-100">Country <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <select name="countryId" id="countryId" class="w-full px-4 py-2.5 rounded-xl border {{ $errors->has('countryId') ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    <option value="" disabled {{ old('countryId') === null ? 'selected' : '' }}>Select Country</option>
                                    @foreach($regions as $region)
                                        <option value="{{ $region->id }}" {{ old('countryId') == $region->id ? 'selected' : '' }}>{{ $region->name }}</option>
                                    @endforeach
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @error('countryId') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                        {{-- Profile Photo --}}
                        <div class="space-y-1">
                            <label for="profilePhoto" class="text-sm font-bold text-primary-100">Profile Photo</label>
                            <div class="relative">
                                <input type="file" name="profilePhoto" id="profilePhoto" accept="image/*"
                                    class="w-full text-sm text-shadedOfGray-60 file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-bold file:bg-primary-10 file:text-primary-100 hover:file:bg-primary-20 cursor-pointer border {{ $errors->has('profilePhoto') ? 'border-red-500' : 'border-shadedOfGray-20' }} rounded-xl bg-primary-10/30">
                            </div>
                             @error('profilePhoto') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                        </div>

                    </div>

                    {{-- Footer Buttons --}}
                    <div class="mt-8 flex items-center justify-end gap-3 border-t border-shadedOfGray-10 pt-6">
                        <button type="button" onclick="toggleCreateModal()" class="px-6 py-2.5 rounded-xl border border-shadedOfGray-20 text-shadedOfGray-60 font-bold hover:bg-shadedOfGray-10 transition-colors text-sm">Cancel</button>
                        <button type="submit" class="px-6 py-2.5 rounded-xl bg-accent-100 text-white font-bold hover:bg-accent-85 shadow-lg shadow-accent-50/50 transition-all text-sm flex items-center gap-2">
                            <i class="fa-solid fa-check"></i> Create User
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    {{-- EDIT USER MODAL --}}
    <div id="editUserModal" class="hidden fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
        <div class="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            <div class="fixed inset-0 bg-primary-100/75 transition-opacity" aria-hidden="true" onclick="toggleEditModal()"></div>
            <span class="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

            <div class="relative inline-block align-bottom bg-white rounded-2xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-3xl w-full">
                {{-- Header --}}
                <div class="bg-primary-100 px-6 py-4 flex justify-between items-center">
                    <h3 class="text-lg font-bold text-white font-primary tracking-wide">Edit User</h3>
                    <button onclick="toggleEditModal()" class="text-primary-30 hover:text-white transition-colors"><i class="fa-solid fa-xmark text-xl"></i></button>
                </div>

                {{-- Form --}}
                <form id="editUserForm" action="#" method="POST" enctype="multipart/form-data" class="p-6 lg:p-8">
                    @csrf
                    @method('PUT')

                    {{-- Hidden input untuk menyimpan ID saat error validation reload --}}
                    <input type="hidden" id="edit_user_id" name="user_id" value="{{ old('user_id') }}">

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">

                        {{-- SECTION: ACCOUNT INFO --}}
                        <div class="md:col-span-2">
                            <p class="text-xs font-bold text-accent-100 uppercase tracking-wider mb-1">Account Information</p>
                            <div class="h-0.5 w-full bg-shadedOfGray-10 mb-4"></div>
                        </div>

                        {{-- Username --}}
                        <div class="space-y-1">
                            <label for="edit_username" class="text-sm font-bold text-primary-100">Username <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-at"></i></span>
                                <input type="text" name="username" id="edit_username" required maxlength="25" value="{{ old('username') }}"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border {{ $errors->has('username') && old('_method') == 'PUT' ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                            @if(old('_method') == 'PUT')
                                @error('username') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                        {{-- Email --}}
                        <div class="space-y-1">
                            <label for="edit_email" class="text-sm font-bold text-primary-100">Email Address <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-regular fa-envelope"></i></span>
                                <input type="email" name="email" id="edit_email" required maxlength="255" value="{{ old('email') }}"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border {{ $errors->has('email') && old('_method') == 'PUT' ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                            @if(old('_method') == 'PUT')
                                @error('email') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                        {{-- Password (Optional) --}}
                        <div class="space-y-1">
                            <label for="edit_password" class="text-sm font-bold text-primary-100">Password <span class="text-shadedOfGray-60 font-normal text-xs">(Leave blank to keep current)</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-lock"></i></span>
                                <input type="password" name="password" id="edit_password" placeholder="••••••••"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border {{ $errors->has('password') && old('_method') == 'PUT' ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                            @if(old('_method') == 'PUT')
                                @error('password') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                        {{-- Role --}}
                        <div class="space-y-1">
                            <label for="edit_role" class="text-sm font-bold text-primary-100">Role <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-shield-halved"></i></span>
                                <select name="role" id="edit_role" class="w-full pl-10 pr-4 py-2.5 rounded-xl border {{ $errors->has('role') && old('_method') == 'PUT' ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    <option value="0">User (Listener)</option>
                                    <option value="1">Admin</option>
                                    <option value="2">Superadmin</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @if(old('_method') == 'PUT')
                                @error('role') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                        {{-- SECTION: PERSONAL INFO --}}
                        <div class="md:col-span-2 mt-2">
                            <p class="text-xs font-bold text-accent-100 uppercase tracking-wider mb-1">Personal Details</p>
                            <div class="h-0.5 w-full bg-shadedOfGray-10 mb-4"></div>
                        </div>

                        {{-- Fullname --}}
                        <div class="space-y-1 md:col-span-2">
                            <label for="edit_fullname" class="text-sm font-bold text-primary-100">Full Name <span class="text-red-500">*</span></label>
                            <input type="text" name="fullname" id="edit_fullname" required maxlength="255" value="{{ old('fullname') }}"
                                class="w-full px-4 py-2.5 rounded-xl border {{ $errors->has('fullname') && old('_method') == 'PUT' ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            @if(old('_method') == 'PUT')
                                @error('fullname') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                        {{-- Date of Birth --}}
                        <div class="space-y-1">
                            <label for="edit_dateOfBirth" class="text-sm font-bold text-primary-100">Date of Birth <span class="text-red-500">*</span></label>
                            <input type="date" name="dateOfBirth" id="edit_dateOfBirth" required value="{{ old('dateOfBirth') }}"
                                class="w-full px-4 py-2.5 rounded-xl border {{ $errors->has('dateOfBirth') && old('_method') == 'PUT' ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            @if(old('_method') == 'PUT')
                                @error('dateOfBirth') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                        {{-- Gender --}}
                        <div class="space-y-1">
                            <label for="edit_gender" class="text-sm font-bold text-primary-100">Gender <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <select name="gender" id="edit_gender" required class="w-full px-4 py-2.5 rounded-xl border {{ $errors->has('gender') && old('_method') == 'PUT' ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    <option value="1">Male</option>
                                    <option value="0">Female</option>
                                    <option value="">Prefer not to say</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @if(old('_method') == 'PUT')
                                @error('gender') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                        {{-- Country --}}
                        <div class="space-y-1">
                            <label for="edit_countryId" class="text-sm font-bold text-primary-100">Country <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <select name="countryId" id="edit_countryId" required class="w-full px-4 py-2.5 rounded-xl border {{ $errors->has('countryId') && old('_method') == 'PUT' ? 'border-red-500 bg-red-50 text-red-900' : 'border-shadedOfGray-20 bg-primary-10/30 text-primary-100' }} focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    @foreach($regions as $region)
                                        <option value="{{ $region->id }}">{{ $region->name }}</option>
                                    @endforeach
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @if(old('_method') == 'PUT')
                                @error('countryId') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                        {{-- Profile Photo --}}
                        <div class="space-y-1">
                            <label for="edit_profilePhoto" class="text-sm font-bold text-primary-100">Profile Photo</label>
                            <div class="relative">
                                <input type="file" name="profilePhoto" id="edit_profilePhoto" accept="image/*"
                                    class="w-full text-sm text-shadedOfGray-60 file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-bold file:bg-primary-10 file:text-primary-100 hover:file:bg-primary-20 cursor-pointer border {{ $errors->has('profilePhoto') && old('_method') == 'PUT' ? 'border-red-500' : 'border-shadedOfGray-20' }} rounded-xl bg-primary-10/30">
                            </div>
                            @if(old('_method') == 'PUT')
                                @error('profilePhoto') <p class="text-red-500 text-xs mt-1 font-medium error-msg">{{ $message }}</p> @enderror
                            @endif
                        </div>

                    </div>

                    {{-- Footer Buttons --}}
                    <div class="mt-8 flex items-center justify-end gap-3 border-t border-shadedOfGray-10 pt-6">
                        <button type="button" onclick="toggleEditModal()" class="px-6 py-2.5 rounded-xl border border-shadedOfGray-20 text-shadedOfGray-60 font-bold hover:bg-shadedOfGray-10 transition-colors text-sm">Cancel</button>
                        <button type="submit" class="px-6 py-2.5 rounded-xl bg-accent-100 text-white font-bold hover:bg-accent-85 shadow-lg shadow-accent-50/50 transition-all text-sm flex items-center gap-2">
                            <i class="fa-solid fa-floppy-disk"></i> Save Changes
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    {{-- DELETE USER MODAL --}}
    <div id="deleteUserModal" class="hidden fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
        <div class="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            {{-- Backdrop --}}
            <div class="fixed inset-0 bg-primary-100/75 transition-opacity" aria-hidden="true" onclick="toggleDeleteModal()"></div>

            <span class="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

            {{-- Modal Content --}}
            <div class="relative inline-block align-bottom bg-white rounded-2xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-md w-full">

                {{-- Header (Merah) --}}
                <div class="bg-secondary-angry-100 px-6 py-4 flex justify-between items-center">
                    <h3 class="text-lg font-bold text-white font-primary tracking-wide">Delete User</h3>
                    <button onclick="toggleDeleteModal()" class="text-white/70 hover:text-white transition-colors">
                        <i class="fa-solid fa-xmark text-xl"></i>
                    </button>
                </div>

                {{-- Body --}}
                <div class="p-6">
                    <div class="flex items-center gap-4 mb-4">
                        <div class="w-12 h-12 rounded-full bg-secondary-angry-10 flex items-center justify-center flex-shrink-0">
                            <i class="fa-solid fa-triangle-exclamation text-secondary-angry-100 text-xl"></i>
                        </div>
                        <div>
                            <h4 class="font-bold text-primary-100 text-lg">Are you sure?</h4>
                            <p class="text-sm text-shadedOfGray-60 mt-1">
                                Do you really want to delete user <span id="delete_username_display" class="font-bold text-primary-100"></span>?
                                This process cannot be undone.
                            </p>
                        </div>
                    </div>

                    {{-- Form Delete --}}
                    <form id="deleteUserForm" action="#" method="POST">
                        @csrf
                        @method('DELETE')

                        <div class="flex items-center justify-end gap-3 mt-6">
                            <button type="button" onclick="toggleDeleteModal()" class="px-5 py-2.5 rounded-xl border border-shadedOfGray-20 text-shadedOfGray-60 font-bold hover:bg-shadedOfGray-10 transition-colors text-sm">
                                Cancel
                            </button>
                            <button type="submit" class="px-5 py-2.5 rounded-xl bg-secondary-angry-100 text-white font-bold hover:bg-secondary-angry-85 shadow-lg shadow-secondary-angry-50/50 transition-all text-sm flex items-center gap-2">
                                <i class="fa-solid fa-trash-can"></i> Delete
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>

    {{-- SCRIPTS --}}
    <script>
        // --- CREATE MODAL LOGIC ---
        function toggleCreateModal() {
            const modal = document.getElementById('createUserModal');
            const form = document.getElementById('createUserForm');

            if (modal.classList.contains('hidden')) {
                modal.classList.remove('hidden');
                document.body.style.overflow = 'hidden';
            } else {
                modal.classList.add('hidden');
                document.body.style.overflow = 'auto';

                // Cleanup Create Form only if not submitting
                if(form) {
                    form.reset();
                    clearErrors(form);
                }
            }
        }

        // --- EDIT MODAL LOGIC ---
        function toggleEditModal() {
            const modal = document.getElementById('editUserModal');
            const form = document.getElementById('editUserForm');

            if (modal.classList.contains('hidden')) {
                modal.classList.remove('hidden');
                document.body.style.overflow = 'hidden';
            } else {
                modal.classList.add('hidden');
                document.body.style.overflow = 'auto';
                // Cleanup Edit Form
                if(form) {
                    form.reset();
                    clearErrors(form);
                }
            }
        }

        // Function to Populate and Open Edit Modal
        function openEditModal(user, profilePhotoUrl) {
            // 1. Set Action URL dynamically
            const form = document.getElementById('editUserForm');
            // Pastikan user.id ada, gunakan fallback jika perlu
            form.action = "{{ route('superadmin.users.index') }}/" + user.id;

            // 2. Set ID for Error Handling
            document.getElementById('edit_user_id').value = user.id;

            // 3. Populate Fields (Tabel Users Utama)
            document.getElementById('edit_username').value = user.username;
            document.getElementById('edit_role').value = user.role;

            // 4. Populate Fields dari User Detail (Tabel Relasi)
            // Laravel mengubah nama relasi 'userDetail' menjadi 'user_detail' (snake_case) di JSON
            if(user.user_detail) {
                // Fix: Ambil email dari user_detail, bukan root user
                document.getElementById('edit_email').value = user.user_detail.email;

                document.getElementById('edit_fullname').value = user.user_detail.fullname;

                // Fix: Ambil DOB dari user_detail
                let dob = user.user_detail.dateOfBirth;
                document.getElementById('edit_dateOfBirth').value = dob ? dob.substring(0, 10) : '';

                // Handle Gender
                let genderVal = user.user_detail.gender;
                if(genderVal === null) genderVal = "";
                document.getElementById('edit_gender').value = genderVal;

                document.getElementById('edit_countryId').value = user.user_detail.countryId;
            } else {
                // Reset jika user_detail kosong (opsional)
                document.getElementById('edit_email').value = '';
                document.getElementById('edit_fullname').value = '';
                document.getElementById('edit_dateOfBirth').value = '';
            }

            // 5. Open the modal
            toggleEditModal();
        }

        // Helper to clear error styles
        function clearErrors(form) {
            const errorInputs = form.querySelectorAll('.border-red-500');
            errorInputs.forEach(input => {
                input.classList.remove('border-red-500', 'bg-red-50', 'text-red-900');
                input.classList.add('border-shadedOfGray-20', 'bg-primary-10/30', 'text-primary-100');
            });

            // Clear text values only for non-hidden inputs
            const inputs = form.querySelectorAll('input:not([type="hidden"]):not([name="_token"]):not([name="_method"]), select');
            inputs.forEach(input => input.value = '');

            const errorMsgs = form.querySelectorAll('.error-msg');
            errorMsgs.forEach(msg => msg.remove());
        }

        // --- AUTO OPEN MODAL ON ERROR ---
        @if($errors->any())
            document.addEventListener('DOMContentLoaded', function() {
                // Check which method was used to decide which modal to reopen
                // Laravel stores spoofed method in old('_method')
                const method = "{{ old('_method') }}";

                if (method === 'PUT') {
                    // Re-open EDIT Modal
                    const modal = document.getElementById('editUserModal');
                    const form = document.getElementById('editUserForm');
                    const userId = document.getElementById('edit_user_id').value;

                    // Re-set action URL because page reloaded
                    if(userId) {
                        form.action = "{{ route('superadmin.users.index') }}/" + userId;
                    }

                    modal.classList.remove('hidden');
                    document.body.style.overflow = 'hidden';
                } else {
                    // Re-open CREATE Modal (Default for POST)
                    const modal = document.getElementById('createUserModal');
                    modal.classList.remove('hidden');
                    document.body.style.overflow = 'hidden';
                }
            });
        @endif

        // --- FILTER & POPUP LOGIC ---
        function toggleFilter() {
            const popup = document.getElementById('filterPopup');
            if (popup.classList.contains('hidden')) {
                popup.classList.remove('hidden');
            } else {
                popup.classList.add('hidden');
            }
        }

        document.addEventListener('click', function(event) {
            const popup = document.getElementById('filterPopup');
            const btn = document.getElementById('filterBtn');
            if (!popup.contains(event.target) && !btn.contains(event.target)) {
                popup.classList.add('hidden');
            }
        });

        // --- DELETE MODAL LOGIC ---
        function toggleDeleteModal() {
            const modal = document.getElementById('deleteUserModal');
            if (modal.classList.contains('hidden')) {
                modal.classList.remove('hidden');
                document.body.style.overflow = 'hidden';
            } else {
                modal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }
        }

        function openDeleteModal(userId, username) {
            // 1. Set Action URL
            const form = document.getElementById('deleteUserForm');
            let url = "{{ route('superadmin.users.destroy', ':id') }}";

            url = url.replace(':id', userId);

            form.action = url;

            // 2. Tampilkan username di pesan konfirmasi
            document.getElementById('delete_username_display').textContent = "@" + username;

            // 3. Buka Modal
            toggleDeleteModal();
        }
    </script>
@endsection
