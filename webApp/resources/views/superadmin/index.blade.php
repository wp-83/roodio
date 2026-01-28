@extends('layouts.superadmin.master')

@section('title', 'User Management')
@section('page_title', 'User Management')
@section('page_subtitle', 'Manage system users permissions')

@section('content')

    {{-- 1. FLASH MESSAGE (Alpine.js Style) --}}
    @if(session('success'))
        <div class="mb-6 bg-secondary-relaxed-100/10 border border-secondary-relaxed-100/20 text-secondary-relaxed-100 px-5 py-4 rounded-xl relative shadow-lg flex items-center gap-3 animate-fade-in-down">
            <i class="fa-solid fa-circle-check text-xl"></i>
            <span class="block sm:inline font-medium">{{ session('success') }}</span>
            <button onclick="this.parentElement.remove()" class="absolute top-0 bottom-0 right-0 px-4 py-3 hover:text-white transition-colors">
                <i class="fa-solid fa-xmark"></i>
            </button>
        </div>
    @endif

    {{-- 2. STATS CARDS --}}
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-6 mb-8">
        {{-- Card 1: Total Users --}}
        <div class="relative overflow-hidden bg-gradient-to-br from-primary-85 to-primary-100 p-6 rounded-2xl shadow-lg border border-primary-70 flex items-center justify-between group">
            <div class="relative z-10">
                <p class="text-sm text-shadedOfGray-40 mb-1 font-secondaryAndButton font-bold uppercase tracking-wider">Total Users</p>
                <h3 class="font-primary text-2xl lg:text-3xl text-white font-bold">{{ number_format($totalUser) }}</h3>
            </div>
            <div class="relative z-10 w-12 h-12 lg:w-14 lg:h-14 rounded-xl bg-secondary-happy-20 flex items-center justify-center text-secondary-happy-100 text-xl lg:text-2xl shadow-md group-hover:scale-110 transition-transform">
                <i class="fa-solid fa-users"></i>
            </div>
            <div class="absolute -bottom-6 -right-6 w-24 h-24 bg-secondary-happy-100/10 rounded-full blur-2xl group-hover:bg-secondary-happy-100/20 transition-colors"></div>
        </div>

        {{-- Card 2: New Registrations --}}
        <div class="relative overflow-hidden bg-gradient-to-br from-primary-85 to-primary-100 p-6 rounded-2xl shadow-lg border border-primary-70 flex items-center justify-between group">
            <div class="relative z-10">
                <p class="text-sm text-shadedOfGray-40 mb-1 font-secondaryAndButton font-bold uppercase tracking-wider">New Registrations</p>
                <h3 class="font-primary text-2xl lg:text-3xl text-white font-bold">{{ number_format($totalNewUser) }}</h3>
            </div>
            <div class="relative z-10 w-12 h-12 lg:w-14 lg:h-14 rounded-xl bg-accent-20 flex items-center justify-center text-accent-100 text-xl lg:text-2xl shadow-md group-hover:scale-110 transition-transform">
                <i class="fa-solid fa-user-plus"></i>
            </div>
            <div class="absolute -bottom-6 -right-6 w-24 h-24 bg-accent-100/10 rounded-full blur-2xl group-hover:bg-accent-100/20 transition-colors"></div>
        </div>
    </div>

    {{-- 3. TABLE CONTAINER --}}
    <div class="bg-primary-85 rounded-2xl shadow-lg border border-primary-70 overflow-visible">

        {{-- TOOLBAR --}}
        <div class="p-4 lg:p-6 border-b border-primary-70 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-primary-85/50 backdrop-blur-sm rounded-t-2xl">

            {{-- SEARCH FORM --}}
            <form action="{{ route('superadmin.users.index') }}" method="GET" class="w-full md:w-80 lg:w-96">
                @if(request()->has('role'))
                    <input type="hidden" name="role" value="{{ request('role') }}">
                @endif
                <div class="relative w-full group">
                    <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40 group-focus-within:text-secondary-happy-100 transition-colors pointer-events-none">
                        <i class="fa-solid fa-magnifying-glass"></i>
                    </span>
                    <input type="text" name="search" value="{{ request('search') }}" placeholder="Search by username..."
                        class="w-full pl-10 pr-10 py-2.5 lg:py-3 rounded-xl border border-primary-60 bg-primary-100 text-white focus:outline-none focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-all shadow-md placeholder-shadedOfGray-40">

                    @if(request()->filled('search'))
                        <a href="{{ route('superadmin.users.index', request()->except(['search', 'page'])) }}" class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 hover:text-white cursor-pointer transition-colors" title="Clear Search">
                            <i class="fa-solid fa-xmark"></i>
                        </a>
                    @endif
                    <button type="submit" class="hidden"></button>
                </div>
            </form>

            <div class="flex gap-2 lg:gap-3 w-full md:w-auto">
                {{-- FILTER BUTTON & POPUP --}}
                <div class="relative flex-1 md:flex-none">
                    <button onclick="toggleFilter()" id="filterBtn" class="w-full px-4 lg:px-5 py-2.5 lg:py-3 rounded-xl border border-primary-60 text-shadedOfGray-30 font-bold hover:bg-primary-70 hover:text-white transition-colors flex items-center justify-center gap-2 text-sm shadow-md">
                        <i class="fa-solid fa-filter"></i>
                        <span class="hidden sm:inline">Filter</span>
                        @if(request()->has('role'))
                            <span class="w-2 h-2 rounded-full bg-accent-100 ml-1"></span>
                        @endif
                    </button>

                    {{-- Filter Dropdown --}}
                    <div id="filterPopup" class="hidden absolute top-full right-0 mt-2 w-48 bg-primary-85 rounded-xl shadow-2xl border border-primary-70 z-50 overflow-hidden">
                        <div class="px-4 py-3 border-b border-primary-70 bg-primary-70/30">
                            <p class="text-xs font-bold text-shadedOfGray-40 uppercase tracking-wider">Filter by Role</p>
                        </div>
                        <div class="py-2 flex flex-col">
                            <a href="{{ route('superadmin.users.index', ['role' => 0]) }}" class="px-4 py-2.5 text-sm hover:bg-primary-70 transition-colors flex items-center justify-between {{ request('role') === '0' ? 'text-accent-100 font-bold bg-accent-100/10' : 'text-shadedOfGray-20' }}">
                                <span>User</span> @if(request('role') === '0') <i class="fa-solid fa-check text-xs"></i> @endif
                            </a>
                            <a href="{{ route('superadmin.users.index', ['role' => 1]) }}" class="px-4 py-2.5 text-sm hover:bg-primary-70 transition-colors flex items-center justify-between {{ request('role') === '1' ? 'text-accent-100 font-bold bg-accent-100/10' : 'text-shadedOfGray-20' }}">
                                <span>Admin</span> @if(request('role') === '1') <i class="fa-solid fa-check text-xs"></i> @endif
                            </a>
                            <a href="{{ route('superadmin.users.index', ['role' => 2]) }}" class="px-4 py-2.5 text-sm hover:bg-primary-70 transition-colors flex items-center justify-between {{ request('role') === '2' ? 'text-accent-100 font-bold bg-accent-100/10' : 'text-shadedOfGray-20' }}">
                                <span>Superadmin</span> @if(request('role') === '2') <i class="fa-solid fa-check text-xs"></i> @endif
                            </a>
                        </div>
                        @if(request()->has('role'))
                            <div class="border-t border-primary-70">
                                <a href="{{ route('superadmin.users.index') }}" class="block px-4 py-3 text-xs text-center text-shadedOfGray-40 hover:text-white font-bold transition-colors bg-primary-70/20">Reset Filter</a>
                            </div>
                        @endif
                    </div>
                </div>

                <button onclick="toggleCreateModal()" class="flex-1 md:flex-none bg-secondary-happy-100 hover:bg-secondary-happy-85 text-white px-4 lg:px-6 py-2.5 lg:py-3 rounded-xl font-bold shadow-lg shadow-secondary-happy-100/20 transition-all flex items-center justify-center gap-2 text-sm transform hover:-translate-y-0.5">
                    <i class="fa-solid fa-plus"></i> <span class="whitespace-nowrap">Add User</span>
                </button>
            </div>
        </div>

        {{-- TABLE WRAPPER --}}
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse min-w-[800px]">
                <thead class="bg-primary-100/50 text-shadedOfGray-30 text-xs uppercase tracking-wider border-b border-primary-70">
                    <tr>
                        <th class="px-6 py-4 font-bold">User</th>
                        <th class="px-6 py-4 font-bold">Role</th>
                        <th class="px-6 py-4 font-bold">Joined Date</th>
                        <th class="px-6 py-4 font-bold text-right">Actions</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-primary-70 text-sm">
                    @foreach($users as $user)
                        <tr class="hover:bg-primary-70/30 transition-colors duration-150 group">
                            <td class="px-6 py-4">
                                <div class="flex items-center gap-3">
                                    {{-- Avatar --}}
                                    <div class="flex-shrink-0">
                                        @if ($user->userDetail?->profilePhoto)
                                            <img class="w-10 h-10 rounded-full object-cover border border-primary-60" src="{{ config('filesystems.disks.azure.url') . '/' . $user->userDetail?->profilePhoto }}" alt="{{ $user->userDetail?->fullname }}">
                                        @else
                                            <div class='w-10 h-10 rounded-full object-cover font-primary font-bold flex items-center justify-center bg-primary-60 text-white border border-primary-50'>
                                                {{ Str::length($user->username) > 0 ? Str::upper(Str::substr($user->username, 0, 1)) : '?' }}
                                            </div>
                                        @endif
                                    </div>
                                    {{-- Info --}}
                                    <div>
                                        <p class="text-white font-bold group-hover:text-secondary-happy-100 transition-colors">{{ $user->userDetail?->fullname ?? 'No Name' }}</p>
                                        <p class="text-xs text-shadedOfGray-40"><span>@</span>{{ $user->username }}</p>
                                    </div>
                                </div>
                            </td>
                            <td class="px-6 py-4">
                                @php
                                    $roleConfig = match($user->role) {
                                        0 => ['name' => 'User', 'class' => 'bg-primary-70 text-shadedOfGray-30 border border-primary-60'],
                                        1 => ['name' => 'Admin', 'class' => 'bg-accent-100/10 text-accent-100 border border-accent-100/20'],
                                        2 => ['name' => 'Superadmin', 'class' => 'bg-secondary-happy-100/10 text-secondary-happy-100 border border-secondary-happy-100/20'],
                                        default => ['name' => 'Unknown', 'class' => 'bg-gray-700 text-gray-300']
                                    };
                                @endphp
                                <span class="inline-flex items-center px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wide {{ $roleConfig['class'] }}">
                                    {{ $roleConfig['name'] }}
                                </span>
                            </td>
                            <td class="px-6 py-4 text-shadedOfGray-40 font-mono text-xs">{{ $user->created_at->format('d M Y') }}</td>
                            <td class="px-6 py-4 text-right">
                                <div class="flex items-center justify-end gap-2 group-hover:opacity-100 transition-opacity">
                                    <button onclick="openEditModal({{ json_encode($user) }}, '{{ $user->userDetail?->profilePhoto }}')"
                                        class="w-8 h-8 rounded-lg flex items-center justify-center text-accent-100 hover:bg-accent-100/10 border border-transparent hover:border-accent-100/20 transition-all shadow-sm">
                                        <i class="fa-solid fa-pen-to-square"></i>
                                    </button>
                                    <button onclick="openDeleteModal('{{ $user->id }}', '{{ $user->username }}')"
                                        class="w-8 h-8 rounded-lg flex items-center justify-center text-secondary-angry-100 hover:bg-secondary-angry-100/10 border border-transparent hover:border-secondary-angry-100/20 transition-all shadow-sm"
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
        <div class="p-4 lg:p-6 border-t border-primary-70 flex flex-col sm:flex-row items-center justify-between gap-4 bg-primary-85/50 rounded-b-2xl">
            <p class="text-sm text-shadedOfGray-40 text-center sm:text-left">
                Showing <span class="font-bold text-white">{{ $users->firstItem() ?? 0 }}</span> to <span class="font-bold text-white">{{ $users->lastItem() ?? 0 }}</span> of <span class="font-bold text-white">{{ $users->total() }}</span> users
            </p>
            <div>{{ $users->appends(request()->query())->links('pagination.superadmin') }}</div>
        </div>
    </div>

    {{-- 4. CREATE USER MODAL (Content Centered) --}}
    <div id="createUserModal" class="hidden fixed inset-0 lg:left-72 lg:top-20 z-50 flex items-center justify-center transition-all duration-300">

        {{-- Backdrop (Gelap hanya di area konten) --}}
        <div class="absolute inset-0 bg-[#020a36]/80 backdrop-blur-sm" onclick="toggleCreateModal()"></div>

        {{-- Modal Panel --}}
        <div class="relative bg-primary-85 rounded-2xl shadow-2xl border border-primary-70 w-full max-w-3xl m-4 flex flex-col max-h-[90%]">

            {{-- Header --}}
            <div class="px-6 py-4 border-b border-primary-70 flex justify-between items-center bg-primary-85 rounded-t-2xl shrink-0">
                <h3 class="text-lg font-bold text-white font-primary tracking-wide">Create New User</h3>
                <button onclick="toggleCreateModal()" class="text-shadedOfGray-40 hover:text-white transition-colors">
                    <i class="fa-solid fa-xmark text-xl"></i>
                </button>
            </div>

            {{-- Scrollable Content --}}
            <div class="overflow-y-auto p-6 lg:p-8 custom-scrollbar">
                <form id="createUserForm" action="{{ route('superadmin.users.store') }}" method="POST" enctype="multipart/form-data">
                    @csrf

                    @if ($errors->any())
                        <div id="globalErrorBox" class="mb-6 p-4 rounded-xl bg-secondary-angry-100/10 border border-secondary-angry-100/20">
                            <p class="text-secondary-angry-100 font-bold text-sm mb-1">Please fix errors:</p>
                            <ul class="list-disc list-inside text-xs text-secondary-angry-100/80">
                                @foreach ($errors->all() as $error) <li>{{ $error }}</li> @endforeach
                            </ul>
                        </div>
                    @endif

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="md:col-span-2">
                            <p class="text-xs font-bold text-secondary-happy-100 uppercase tracking-wider mb-1">Account Information</p>
                            <div class="h-px w-full bg-primary-70 mb-4"></div>
                        </div>

                        {{-- Username --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Username <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-at"></i></span>
                                <input type="text" name="username" maxlength="25" placeholder="e.g. johndoe" value="{{ old('username') }}"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors placeholder-shadedOfGray-40">
                            </div>
                            @error('username') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        {{-- Email --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Email <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-regular fa-envelope"></i></span>
                                <input type="email" name="email" maxlength="255" placeholder="john@example.com" value="{{ old('email') }}"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors placeholder-shadedOfGray-40">
                            </div>
                            @error('email') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        {{-- Password --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Password <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-lock"></i></span>
                                <input type="password" name="password" placeholder="••••••••"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors placeholder-shadedOfGray-40">
                            </div>
                            @error('password') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        {{-- Role --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Role <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-shield-halved"></i></span>
                                <select name="role" class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm appearance-none">
                                    <option value="0" {{ old('role') == '0' ? 'selected' : '' }}>User (Listener)</option>
                                    <option value="1" {{ old('role') == '1' ? 'selected' : '' }}>Admin</option>
                                    <option value="2" {{ old('role') == '2' ? 'selected' : '' }}>Superadmin</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @error('role') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        <div class="md:col-span-2 mt-2">
                            <p class="text-xs font-bold text-secondary-happy-100 uppercase tracking-wider mb-1">Personal Details</p>
                            <div class="h-px w-full bg-primary-70 mb-4"></div>
                        </div>

                        {{-- Fullname --}}
                        <div class="space-y-1 md:col-span-2">
                            <label class="text-sm font-bold text-shadedOfGray-20">Full Name <span class="text-secondary-angry-100">*</span></label>
                            <input type="text" name="fullname" maxlength="255" placeholder="e.g. Johnathan Doe" value="{{ old('fullname') }}"
                                class="w-full px-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors placeholder-shadedOfGray-40">
                            @error('fullname') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        {{-- Date of Birth --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Date of Birth <span class="text-secondary-angry-100">*</span></label>
                            <input type="date" name="dateOfBirth" value="{{ old('dateOfBirth') }}"
                                class="w-full px-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors [color-scheme:dark]">
                            @error('dateOfBirth') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        {{-- Gender --}}
                        <div class="space-y-1">
                            <label for="gender" class="text-sm font-bold text-shadedOfGray-20">Gender <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <select name="gender" id="gender" class="w-full px-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm appearance-none cursor-pointer">

                                    {{-- 1. Placeholder (Terpilih jika old value kosong/null) --}}
                                    <option value="" disabled @selected(old('gender') === null)>Select Gender</option>

                                    {{-- 2. Male --}}
                                    <option value="1" @selected(old('gender') == '1')>Male</option>

                                    {{-- 3. Female (Gunakan string '0' agar aman) --}}
                                    <option value="0" @selected(old('gender') === '0')>Female</option>

                                    {{-- 4. Prefer not to say (Ubah value jadi -1 agar tidak bentrok dengan placeholder) --}}
                                    <option value="-1" @selected(old('gender') == '-1')>Prefer not to say</option>

                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none">
                                    <i class="fa-solid fa-chevron-down text-xs"></i>
                                </span>
                            </div>
                            @error('gender') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        {{-- Country --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Country <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <select name="countryId" class="w-full px-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm appearance-none">
                                    <option value="" disabled {{ old('countryId') === null ? 'selected' : '' }}>Select Country</option>
                                    @foreach($regions as $region)
                                        <option value="{{ $region->id }}" {{ old('countryId') == $region->id ? 'selected' : '' }}>{{ $region->name }}</option>
                                    @endforeach
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @error('countryId') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>

                        {{-- Profile Photo --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Profile Photo</label>
                            <input type="file" name="profilePhoto" accept="image/*"
                                class="w-full text-sm text-shadedOfGray-40 file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-bold file:bg-primary-60 file:text-white hover:file:bg-primary-50 cursor-pointer border border-primary-60 rounded-xl bg-primary-100">
                            @error('profilePhoto') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror
                        </div>
                    </div>

                    {{-- Footer --}}
                    <div class="mt-8 flex items-center justify-end gap-3 pt-6 border-t border-primary-70">
                        <button type="button" onclick="toggleCreateModal()" class="px-6 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-40 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm">Cancel</button>
                        <button type="submit" class="px-6 py-2.5 rounded-xl bg-secondary-happy-100 text-white font-bold hover:bg-secondary-happy-85 shadow-lg shadow-secondary-happy-100/20 transition-all text-sm flex items-center gap-2">
                            <i class="fa-solid fa-check"></i> Create User
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>

   {{-- EDIT USER MODAL (Content Centered) --}}
    <div id="editUserModal" class="hidden fixed inset-0 lg:left-72 lg:top-20 z-50 flex items-center justify-center transition-all duration-300">
        <div class="absolute inset-0 bg-[#020a36]/80 backdrop-blur-sm" onclick="toggleEditModal()"></div>

        <div class="relative bg-primary-85 rounded-2xl shadow-2xl border border-primary-70 w-full max-w-3xl m-4 flex flex-col max-h-[90%]">
            {{-- Header --}}
            <div class="px-6 py-4 border-b border-primary-70 flex justify-between items-center bg-primary-85 rounded-t-2xl shrink-0">
                <h3 class="text-lg font-bold text-white font-primary tracking-wide">Edit User</h3>
                <button onclick="toggleEditModal()" class="text-shadedOfGray-40 hover:text-white transition-colors">
                    <i class="fa-solid fa-xmark text-xl"></i>
                </button>
            </div>

            {{-- Scrollable Content --}}
            <div class="overflow-y-auto p-6 lg:p-8 custom-scrollbar">
                <form id="editUserForm" action="#" method="POST" enctype="multipart/form-data">
                    @csrf
                    @method('PUT')
                    <input type="hidden" id="edit_user_id" name="user_id" value="{{ old('user_id') }}">

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="md:col-span-2">
                            <p class="text-xs font-bold text-secondary-happy-100 uppercase tracking-wider mb-1">Account Information</p>
                            <div class="h-px w-full bg-primary-70 mb-4"></div>
                        </div>

                        {{-- Username --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Username <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-at"></i></span>
                                <input type="text" name="username" id="edit_username" required maxlength="25" value="{{ old('username') }}"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors">
                            </div>
                            @if(old('_method') == 'PUT') @error('username') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>

                        {{-- Email --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Email <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-regular fa-envelope"></i></span>
                                <input type="email" name="email" id="edit_email" required maxlength="255" value="{{ old('email') }}"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors">
                            </div>
                            @if(old('_method') == 'PUT') @error('email') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>

                        {{-- Password --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Password <span class="text-shadedOfGray-60 font-normal text-xs">(Leave blank to keep current)</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-lock"></i></span>
                                <input type="password" name="password" id="edit_password" placeholder="••••••••"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors">
                            </div>
                            @if(old('_method') == 'PUT') @error('password') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>

                        {{-- Role --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Role <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-shield-halved"></i></span>
                                <select name="role" id="edit_role" class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm appearance-none">
                                    <option value="0">User (Listener)</option>
                                    <option value="1">Admin</option>
                                    <option value="2">Superadmin</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @if(old('_method') == 'PUT') @error('role') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>

                        <div class="md:col-span-2 mt-2">
                            <p class="text-xs font-bold text-secondary-happy-100 uppercase tracking-wider mb-1">Personal Details</p>
                            <div class="h-px w-full bg-primary-70 mb-4"></div>
                        </div>

                        {{-- Fullname --}}
                        <div class="space-y-1 md:col-span-2">
                            <label class="text-sm font-bold text-shadedOfGray-20">Full Name <span class="text-secondary-angry-100">*</span></label>
                            <input type="text" name="fullname" id="edit_fullname" required maxlength="255" value="{{ old('fullname') }}"
                                class="w-full px-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors">
                            @if(old('_method') == 'PUT') @error('fullname') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>

                        {{-- Date of Birth --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Date of Birth <span class="text-secondary-angry-100">*</span></label>
                            <input type="date" name="dateOfBirth" id="edit_dateOfBirth" required value="{{ old('dateOfBirth') }}"
                                class="w-full px-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm transition-colors [color-scheme:dark]">
                            @if(old('_method') == 'PUT') @error('dateOfBirth') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>

                        {{-- Gender --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Gender <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <select name="gender" id="edit_gender" required class="w-full px-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm appearance-none">
                                    <option value="1">Male</option>
                                    <option value="0">Female</option>
                                    <option value="">Prefer not to say</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @if(old('_method') == 'PUT') @error('gender') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>

                        {{-- Country --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Country <span class="text-secondary-angry-100">*</span></label>
                            <div class="relative">
                                <select name="countryId" id="edit_countryId" required class="w-full px-4 py-2.5 rounded-xl border border-primary-60 bg-primary-100 text-white focus:border-secondary-happy-100 focus:ring-1 focus:ring-secondary-happy-100 text-sm appearance-none">
                                    @foreach($regions as $region)
                                        <option value="{{ $region->id }}">{{ $region->name }}</option>
                                    @endforeach
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                            @if(old('_method') == 'PUT') @error('countryId') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>

                        {{-- Profile Photo --}}
                        <div class="space-y-1">
                            <label class="text-sm font-bold text-shadedOfGray-20">Profile Photo</label>
                            <input type="file" name="profilePhoto" id="edit_profilePhoto" accept="image/*"
                                class="w-full text-sm text-shadedOfGray-40 file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-bold file:bg-primary-60 file:text-white hover:file:bg-primary-50 cursor-pointer border border-primary-60 rounded-xl bg-primary-100">
                            @if(old('_method') == 'PUT') @error('profilePhoto') <p class="text-secondary-angry-100 text-xs mt-1">{{ $message }}</p> @enderror @endif
                        </div>
                    </div>

                    {{-- Footer --}}
                    <div class="mt-8 flex items-center justify-end gap-3 pt-6 border-t border-primary-70">
                        <button type="button" onclick="toggleEditModal()" class="px-6 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-40 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm">Cancel</button>
                        <button type="submit" id="editSubmitBtn" disabled
                            class="px-6 py-2.5 rounded-xl bg-secondary-happy-100 text-white font-bold shadow-lg shadow-secondary-happy-100/20 transition-all text-sm flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-secondary-happy-85">
                            <i class="fa-solid fa-floppy-disk"></i> Save Changes
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    {{-- DELETE USER MODAL (Content Centered) --}}
    <div id="deleteUserModal" class="hidden fixed inset-0 lg:left-72 lg:top-20 z-50 flex items-center justify-center transition-all duration-300">
        <div class="absolute inset-0 bg-[#020a36]/80 backdrop-blur-sm" onclick="toggleDeleteModal()"></div>

        <div class="relative bg-primary-85 rounded-2xl shadow-2xl border border-primary-70 w-full max-w-md m-4">
            {{-- Header --}}
            <div class="bg-secondary-angry-100 px-6 py-4 flex justify-between items-center rounded-t-2xl">
                <h3 class="text-lg font-bold text-white font-primary tracking-wide">Delete User</h3>
                <button onclick="toggleDeleteModal()" class="text-white/70 hover:text-white transition-colors">
                    <i class="fa-solid fa-xmark text-xl"></i>
                </button>
            </div>

            {{-- Body --}}
            <div class="p-6">
                <div class="flex items-center gap-4 mb-4">
                    <div class="w-12 h-12 rounded-full bg-secondary-angry-100/20 flex items-center justify-center flex-shrink-0 border border-secondary-angry-100/30">
                        <i class="fa-solid fa-triangle-exclamation text-secondary-angry-100 text-xl"></i>
                    </div>
                    <div>
                        <h4 class="font-bold text-white text-lg">Are you sure?</h4>
                        <p class="text-sm text-shadedOfGray-40 mt-1">
                            Do you really want to delete user <span id="delete_username_display" class="font-bold text-white"></span>? This process cannot be undone.
                        </p>
                    </div>
                </div>

                <form id="deleteUserForm" action="#" method="POST">
                    @csrf
                    @method('DELETE')
                    <div class="flex items-center justify-end gap-3 mt-6 pt-4 border-t border-primary-70">
                        <button type="button" onclick="toggleDeleteModal()" class="px-5 py-2.5 rounded-xl border border-primary-60 text-shadedOfGray-40 font-bold hover:bg-primary-70 hover:text-white transition-colors text-sm">Cancel</button>
                        <button type="submit" class="px-5 py-2.5 rounded-xl bg-secondary-angry-100 text-white font-bold hover:bg-secondary-angry-85 shadow-lg shadow-secondary-angry-100/20 transition-all text-sm flex items-center gap-2">
                            <i class="fa-solid fa-trash-can"></i> Delete
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    {{-- SCRIPTS (Sama dengan sebelumnya, hanya logic) --}}
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
                if(form) { form.reset(); clearErrors(form); }
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
                if(form) { form.reset(); clearErrors(form); }
            }
        }

        let initialEditData = {};
        function openEditModal(user, profilePhotoUrl) {
            const form = document.getElementById('editUserForm');
            form.action = "{{ route('superadmin.users.index') }}/" + user.id;
            document.getElementById('edit_user_id').value = user.id;
            document.getElementById('edit_username').value = user.username;
            document.getElementById('edit_role').value = user.role;
            document.getElementById('edit_password').value = '';
            document.getElementById('edit_profilePhoto').value = '';

            if(user.user_detail) {
                document.getElementById('edit_email').value = user.user_detail.email;
                document.getElementById('edit_fullname').value = user.user_detail.fullname;
                let dob = user.user_detail.dateOfBirth;
                document.getElementById('edit_dateOfBirth').value = dob ? dob.substring(0, 10) : '';
                let genderVal = user.user_detail.gender;
                if(genderVal === null) genderVal = "";
                document.getElementById('edit_gender').value = genderVal;
                document.getElementById('edit_countryId').value = user.user_detail.countryId;
            } else {
                ['edit_email', 'edit_fullname', 'edit_dateOfBirth', 'edit_gender', 'edit_countryId'].forEach(id => {
                    document.getElementById(id).value = '';
                });
            }
            saveInitialState();
            setupChangeListener();
            document.getElementById('editSubmitBtn').disabled = true;
            toggleEditModal();
        }

        function saveInitialState() {
            initialEditData = {
                username: document.getElementById('edit_username').value,
                email: document.getElementById('edit_email').value,
                role: document.getElementById('edit_role').value,
                fullname: document.getElementById('edit_fullname').value,
                dob: document.getElementById('edit_dateOfBirth').value,
                gender: document.getElementById('edit_gender').value,
                country: document.getElementById('edit_countryId').value,
            };
        }

        function setupChangeListener() {
            const formInputs = document.querySelectorAll('#editUserForm input, #editUserForm select');
            formInputs.forEach(input => {
                input.removeEventListener('input', checkFormChanges);
                input.removeEventListener('change', checkFormChanges);
                input.addEventListener('input', checkFormChanges);
                input.addEventListener('change', checkFormChanges);
            });
        }

        function checkFormChanges() {
            const currentData = {
                username: document.getElementById('edit_username').value,
                email: document.getElementById('edit_email').value,
                role: document.getElementById('edit_role').value,
                fullname: document.getElementById('edit_fullname').value,
                dob: document.getElementById('edit_dateOfBirth').value,
                gender: document.getElementById('edit_gender').value,
                country: document.getElementById('edit_countryId').value,
            };
            const passwordFilled = document.getElementById('edit_password').value.length > 0;
            const photoFilled = document.getElementById('edit_profilePhoto').files.length > 0;
            const isDataChanged = JSON.stringify(initialEditData) !== JSON.stringify(currentData);
            const btn = document.getElementById('editSubmitBtn');
            if (isDataChanged || passwordFilled || photoFilled) {
                btn.disabled = false;
                btn.classList.remove('opacity-50', 'cursor-not-allowed');
            } else {
                btn.disabled = true;
                btn.classList.add('opacity-50', 'cursor-not-allowed');
            }
        }

        function clearErrors(form) {
            const errorInputs = form.querySelectorAll('.border-secondary-angry-100');
            errorInputs.forEach(input => {
                input.classList.remove('border-secondary-angry-100', 'bg-secondary-angry-100/10');
                input.classList.add('border-primary-60', 'bg-primary-100');
            });
            const inputs = form.querySelectorAll('input:not([type="hidden"]):not([name="_token"]):not([name="_method"]), select');
            inputs.forEach(input => input.value = '');
            const errorMsgs = form.querySelectorAll('.error-msg');
            errorMsgs.forEach(msg => msg.remove());
        }

        @if($errors->any())
            document.addEventListener('DOMContentLoaded', function() {
                const method = "{{ old('_method') }}";
                if (method === 'PUT') {
                    const modal = document.getElementById('editUserModal');
                    const form = document.getElementById('editUserForm');
                    const userId = document.getElementById('edit_user_id').value;
                    if(userId) form.action = "{{ route('superadmin.users.index') }}/" + userId;
                    modal.classList.remove('hidden');
                    document.body.style.overflow = 'hidden';
                } else {
                    const modal = document.getElementById('createUserModal');
                    modal.classList.remove('hidden');
                    document.body.style.overflow = 'hidden';
                }
            });
        @endif

        function toggleFilter() {
            const popup = document.getElementById('filterPopup');
            popup.classList.toggle('hidden');
        }
        document.addEventListener('click', function(event) {
            const popup = document.getElementById('filterPopup');
            const btn = document.getElementById('filterBtn');
            if (!popup.contains(event.target) && !btn.contains(event.target)) {
                popup.classList.add('hidden');
            }
        });

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
            const form = document.getElementById('deleteUserForm');
            let url = "{{ route('superadmin.users.destroy', ':id') }}";
            url = url.replace(':id', userId);
            form.action = url;
            document.getElementById('delete_username_display').textContent = "@" + username;
            toggleDeleteModal();
        }
    </script>
@endsection
