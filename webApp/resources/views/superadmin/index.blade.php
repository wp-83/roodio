@extends('layouts.superadmin.master')

@section('title', 'User Management')
@section('page_title', 'User Management')
@section('page_subtitle', 'Manage system users permissions')

@section('content')
    {{-- STATS CARDS --}}
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

    {{-- TABLE CONTAINER --}}
    <div class="bg-white rounded-2xl shadow-lg border border-shadedOfGray-10 overflow-visible"> {{-- Ubah overflow-hidden jadi visible agar popup tidak terpotong --}}

        {{-- TOOLBAR --}}
        <div class="p-4 lg:p-6 border-b border-shadedOfGray-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
            {{-- SEARCH FORM --}}
            <form action="{{ route('superadmin.users.index') }}" method="GET" class="w-full md:w-80 lg:w-96">

                {{-- PENTING: Pertahankan Filter Role jika ada --}}
                @if(request()->has('role'))
                    <input type="hidden" name="role" value="{{ request('role') }}">
                @endif

                <div class="relative w-full">
                    {{-- Ikon Search (Kiri) --}}
                    <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40 pointer-events-none">
                        <i class="fa-solid fa-magnifying-glass"></i>
                    </span>

                    {{-- Input Search --}}
                    {{-- Perhatikan: saya ubah pr-4 menjadi pr-10 agar teks tidak menabrak tombol X --}}
                    <input
                        type="text"
                        name="search"
                        value="{{ request('search') }}"
                        placeholder="Search by username..."
                        class="w-full pl-10 pr-10 py-2.5 lg:py-3 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm transition-all"
                    >

                    {{-- Tombol Clear / Batal (Kanan) --}}
                    {{-- Hanya muncul jika ada request search --}}
                    @if(request()->filled('search'))
                        <a href="{{ route('superadmin.users.index', request()->except(['search', 'page'])) }}"
                        class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 hover:text-accent-100 cursor-pointer transition-colors"
                        title="Clear Search">
                            <i class="fa-solid fa-xmark"></i>
                        </a>
                    @endif

                    {{-- Tombol Submit Tersembunyi (Agar bisa tekan Enter) --}}
                    <button type="submit" class="hidden"></button>
                </div>
            </form>

            <div class="flex gap-2 lg:gap-3 w-full md:w-auto">

                {{-- MODIFIKASI: BUTTON FILTER DENGAN POPUP --}}
                <div class="relative flex-1 md:flex-none">
                    <button onclick="toggleFilter()" id="filterBtn" class="w-full px-4 lg:px-5 py-2.5 lg:py-3 rounded-xl border border-shadedOfGray-20 text-primary-70 font-medium hover:bg-shadedOfGray-10 transition-colors flex items-center justify-center gap-2 text-sm">
                        <i class="fa-solid fa-filter"></i>
                        <span class="hidden sm:inline">Filter</span>
                        {{-- Indikator jika ada filter aktif --}}
                        @if(request()->has('role'))
                            <span class="w-2 h-2 rounded-full bg-accent-100"></span>
                        @endif
                    </button>

                    {{-- POPUP MENU --}}
                    <div id="filterPopup" class="hidden absolute top-full right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-shadedOfGray-10 z-50 overflow-hidden">
                        <div class="px-4 py-3 border-b border-shadedOfGray-10 bg-primary-10/10">
                            <p class="text-xs font-bold text-shadedOfGray-60 uppercase tracking-wider">Filter by Role</p>
                        </div>
                        <div class="py-2 flex flex-col">
                            {{-- Option: User (0) --}}
                            <a href="{{ route('superadmin.users.index', ['role' => 0]) }}"
                               class="px-4 py-2.5 text-sm hover:bg-primary-10/30 transition-colors flex items-center justify-between {{ request('role') === '0' ? 'text-accent-100 font-bold bg-accent-20' : 'text-primary-100' }}">
                                <span>User</span>
                                @if(request('role') === '0') <i class="fa-solid fa-check"></i> @endif
                            </a>

                            {{-- Option: Admin (1) --}}
                            <a href="{{ route('superadmin.users.index', ['role' => 1]) }}"
                               class="px-4 py-2.5 text-sm hover:bg-primary-10/30 transition-colors flex items-center justify-between {{ request('role') === '1' ? 'text-accent-100 font-bold bg-accent-20' : 'text-primary-100' }}">
                                <span>Admin</span>
                                @if(request('role') === '1') <i class="fa-solid fa-check"></i> @endif
                            </a>

                            {{-- Option: Superadmin (2) --}}
                            <a href="{{ route('superadmin.users.index', ['role' => 2]) }}"
                               class="px-4 py-2.5 text-sm hover:bg-primary-10/30 transition-colors flex items-center justify-between {{ request('role') === '2' ? 'text-accent-100 font-bold bg-accent-20' : 'text-primary-100' }}">
                                <span>Superadmin</span>
                                @if(request('role') === '2') <i class="fa-solid fa-check"></i> @endif
                            </a>
                        </div>

                        {{-- Reset Filter --}}
                        @if(request()->has('role'))
                            <div class="border-t border-shadedOfGray-10">
                                <a href="{{ route('superadmin.users.index') }}" class="block px-4 py-3 text-xs text-center text-shadedOfGray-60 hover:text-accent-100 font-bold transition-colors">
                                    Reset Filter
                                </a>
                            </div>
                        @endif
                    </div>
                </div>
                {{-- END MODIFIKASI --}}

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
                    {{-- Loop data user --}}
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
                                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold {{ $roleClass }}">
                                    {{ $roleName }}
                                </span>
                            </td>
                            <td class="px-6 py-4 text-sm text-shadedOfGray-70">{{ $user->created_at->format('d M Y') }}</td>
                            <td class="px-6 py-4 text-right">
                                <div class="flex items-center justify-end gap-2">
                                    <button class="w-8 h-8 rounded-lg flex items-center justify-center text-accent-100 hover:bg-accent-20 transition-colors"><i class="fa-solid fa-pen-to-square"></i></button>
                                    <button class="w-8 h-8 rounded-lg flex items-center justify-center text-secondary-angry-100 hover:bg-secondary-angry-20 transition-colors"><i class="fa-solid fa-trash-can"></i></button>
                                </div>
                            </td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>

        {{-- PAGINATION --}}
        <div class="p-4 lg:p-6 border-t border-shadedOfGray-10 flex flex-col sm:flex-row items-center justify-between gap-4">

            {{-- 1. INFO TEXT (Showing X to Y of Z) --}}
            <p class="text-sm text-shadedOfGray-60 text-center sm:text-left">
                Showing
                <span class="font-bold text-primary-100">{{ $users->firstItem() ?? 0 }}</span>
                to
                <span class="font-bold text-primary-100">{{ $users->lastItem() ?? 0 }}</span>
                of
                <span class="font-bold text-primary-100">{{ $users->total() }}</span>
                users
            </p>

            {{-- 2. PAGINATION LINKS (Memanggil Custom View) --}}
            {{-- appends() berguna agar filter search/role tidak hilang saat ganti halaman --}}
            <div>
                {{ $users->appends(request()->query())->links('pagination.superadmin') }}
            </div>

        </div>
    </div>

        {{-- CREATE USER MODAL --}}
    <div id="createUserModal" class="hidden fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
        {{-- Backdrop (Gelap Transparan) --}}
        <div class="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            <div class="fixed inset-0 bg-primary-100/75 transition-opacity" aria-hidden="true" onclick="toggleCreateModal()"></div>

            <span class="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

            {{-- Modal Content --}}
            <div class="relative inline-block align-bottom bg-white rounded-2xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-3xl w-full">

                {{-- Header --}}
                <div class="bg-primary-100 px-6 py-4 flex justify-between items-center">
                    <h3 class="text-lg font-bold text-white font-primary tracking-wide">Create New User</h3>
                    <button onclick="toggleCreateModal()" class="text-primary-30 hover:text-white transition-colors">
                        <i class="fa-solid fa-xmark text-xl"></i>
                    </button>
                </div>

                {{-- Form --}}
                <form action="" method="POST" enctype="multipart/form-data" class="p-6 lg:p-8">
                    {{-- {{ route('superadmin.users.store') }} --}}
                    @csrf

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
                                <input type="text" name="username" id="username" required maxlength="25" placeholder="e.g. johndoe"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                        </div>

                        {{-- Email --}}
                        <div class="space-y-1">
                            <label for="email" class="text-sm font-bold text-primary-100">Email Address <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-regular fa-envelope"></i></span>
                                <input type="email" name="email" id="email" required maxlength="255" placeholder="john@example.com"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                        </div>

                        {{-- Password --}}
                        <div class="space-y-1">
                            <label for="password" class="text-sm font-bold text-primary-100">Password <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-lock"></i></span>
                                <input type="password" name="password" id="password" required placeholder="••••••••"
                                    class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                            </div>
                        </div>

                        {{-- Role --}}
                        <div class="space-y-1">
                            <label for="role" class="text-sm font-bold text-primary-100">Role <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40"><i class="fa-solid fa-shield-halved"></i></span>
                                <select name="role" id="role" class="w-full pl-10 pr-4 py-2.5 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    <option value="0">User (Listener)</option>
                                    <option value="1">Admin</option>
                                    <option value="2">Superadmin</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                        </div>

                        {{-- SECTION: PERSONAL INFO --}}
                        <div class="md:col-span-2 mt-2">
                            <p class="text-xs font-bold text-accent-100 uppercase tracking-wider mb-1">Personal Details</p>
                            <div class="h-0.5 w-full bg-shadedOfGray-10 mb-4"></div>
                        </div>

                        {{-- Fullname --}}
                        <div class="space-y-1 md:col-span-2">
                            <label for="fullname" class="text-sm font-bold text-primary-100">Full Name <span class="text-red-500">*</span></label>
                            <input type="text" name="fullname" id="fullname" required maxlength="255" placeholder="e.g. Johnathan Doe"
                                class="w-full px-4 py-2.5 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                        </div>

                        {{-- Date of Birth --}}
                        <div class="space-y-1">
                            <label for="dateOfBirth" class="text-sm font-bold text-primary-100">Date of Birth <span class="text-red-500">*</span></label>
                            <input type="date" name="dateOfBirth" id="dateOfBirth" required
                                class="w-full px-4 py-2.5 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm">
                        </div>

                        {{-- Gender --}}
                        <div class="space-y-1">
                            <label for="gender" class="text-sm font-bold text-primary-100">Gender <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <select name="gender" id="gender" required class="w-full px-4 py-2.5 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    <option value="" disabled selected>Select Gender</option>
                                    <option value="1">Male</option>
                                    <option value="0">Female</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                        </div>

                        {{-- Country (Butuh data regions dari controller) --}}
                        <div class="space-y-1">
                            <label for="countryId" class="text-sm font-bold text-primary-100">Country <span class="text-red-500">*</span></label>
                            <div class="relative">
                                <select name="countryId" id="countryId" required class="w-full px-4 py-2.5 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm appearance-none">
                                    <option value="" disabled selected>Select Country</option>
                                    {{-- CONTOH LOOPING --}}
                                    {{-- @foreach($regions as $region) --}}
                                    {{--     <option value="{{ $region->id }}">{{ $region->name }}</option> --}}
                                    {{-- @endforeach --}}
                                    {{-- Placeholder sementara --}}
                                    <option value="ID">Indonesia</option>
                                    <option value="US">United States</option>
                                </select>
                                <span class="absolute inset-y-0 right-0 pr-3 flex items-center text-shadedOfGray-40 pointer-events-none"><i class="fa-solid fa-chevron-down text-xs"></i></span>
                            </div>
                        </div>

                        {{-- Profile Photo --}}
                        <div class="space-y-1">
                            <label for="profilePhoto" class="text-sm font-bold text-primary-100">Profile Photo</label>
                            <div class="relative">
                                <input type="file" name="profilePhoto" id="profilePhoto" accept="image/*"
                                    class="w-full text-sm text-shadedOfGray-60 file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-bold file:bg-primary-10 file:text-primary-100 hover:file:bg-primary-20 cursor-pointer border border-shadedOfGray-20 rounded-xl bg-primary-10/30">
                            </div>
                        </div>

                    </div>

                    {{-- Footer Buttons --}}
                    <div class="mt-8 flex items-center justify-end gap-3 border-t border-shadedOfGray-10 pt-6">
                        <button type="button" onclick="toggleCreateModal()" class="px-6 py-2.5 rounded-xl border border-shadedOfGray-20 text-shadedOfGray-60 font-bold hover:bg-shadedOfGray-10 transition-colors text-sm">
                            Cancel
                        </button>
                        <button type="submit" class="px-6 py-2.5 rounded-xl bg-accent-100 text-white font-bold hover:bg-accent-85 shadow-lg shadow-accent-50/50 transition-all text-sm flex items-center gap-2">
                            <i class="fa-solid fa-check"></i> Create User
                        </button>
                    </div>

                </form>
            </div>
        </div>
    </div>

    {{-- SCRIPT PENDUKUNG (Tumpuk di bawah script filter yang sudah ada) --}}
    <script>
        function toggleCreateModal() {
            const modal = document.getElementById('createUserModal');
            if (modal.classList.contains('hidden')) {
                modal.classList.remove('hidden');
                // Prevent body scroll when modal is open
                document.body.style.overflow = 'hidden';
            } else {
                modal.classList.add('hidden');
                // Enable body scroll
                document.body.style.overflow = 'auto';
            }
        }
    </script>

    {{-- SCRIPT TOGGLE POPUP --}}
    <script>
        function toggleFilter() {
            const popup = document.getElementById('filterPopup');
            if (popup.classList.contains('hidden')) {
                popup.classList.remove('hidden');
            } else {
                popup.classList.add('hidden');
            }
        }

        // Close popup when clicking outside
        document.addEventListener('click', function(event) {
            const popup = document.getElementById('filterPopup');
            const btn = document.getElementById('filterBtn');

            if (!popup.contains(event.target) && !btn.contains(event.target)) {
                popup.classList.add('hidden');
            }
        });
    </script>
@endsection
