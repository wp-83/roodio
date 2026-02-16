@extends('layouts.superadmin.master')

@section('title', 'Roles & Access')
@section('page_title', 'Roles & Permissions')
@section('page_subtitle', 'Manage user roles and system access levels')

@section('content')

    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">

        {{-- SUPERADMIN CARD (Role 2) --}}
        <div class="relative overflow-hidden bg-primary-85 rounded-2xl shadow-lg border border-primary-70 border-t-4 border-t-accent-100 p-6 group hover:shadow-accent-100/10 transition-all duration-300 hover:-translate-y-1">
            {{-- Background Icon --}}
            <div class="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <i class="fa-solid fa-crown text-8xl text-accent-100"></i>
            </div>

            {{-- Icon --}}
            <div class="mb-4">
                <div class="inline-flex p-3 bg-accent-100/10 rounded-xl text-accent-100 border border-accent-100/20">
                    <i class="fa-solid fa-shield-halved text-2xl"></i>
                </div>
            </div>

            {{-- Title & Desc --}}
            <h3 class="font-primary text-xl font-bold text-white mb-1">Superadmin</h3>
            <p class="text-sm text-primary-20 mb-6 font-secondaryAndButton h-10">Full access to all system features, user management, and configuration.</p>

            {{-- Footer (Total Only) --}}
            <div class="border-t border-primary-70 pt-4">
                <div>
                    <p class="text-xs text-primary-20 font-bold uppercase tracking-wider">Total Accounts</p>
                    <p class="font-primary font-bold text-white text-2xl mt-1">{{ $role2 ?? 0 }}</p>
                </div>
            </div>
        </div>

        {{-- ADMIN CARD (Role 1) --}}
        <div class="relative overflow-hidden bg-primary-85 rounded-2xl shadow-lg border border-primary-70 border-t-4 border-t-secondary-happy-100 p-6 group hover:shadow-secondary-happy-100/10 transition-all duration-300 hover:-translate-y-1">
            <div class="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <i class="fa-solid fa-user-gear text-8xl text-secondary-happy-100"></i>
            </div>

            <div class="mb-4">
                <div class="inline-flex p-3 bg-secondary-happy-100/10 rounded-xl text-secondary-happy-100 border border-secondary-happy-100/20">
                    <i class="fa-solid fa-laptop-file text-2xl"></i>
                </div>
            </div>

            <h3 class="font-primary text-xl font-bold text-white mb-1">Admin</h3>
            <p class="text-sm text-primary-20 mb-6 font-secondaryAndButton h-10">Manage content (Songs, Playlists), view users, and day-to-day operations.</p>

            <div class="border-t border-primary-70 pt-4">
                <div>
                    <p class="text-xs text-primary-20 font-bold uppercase tracking-wider">Total Accounts</p>
                    <p class="font-primary font-bold text-white text-2xl mt-1">{{ $role1 ?? 0 }}</p>
                </div>
            </div>
        </div>

        {{-- USER CARD (Role 0) --}}
        <div class="relative overflow-hidden bg-primary-85 rounded-2xl shadow-lg border border-primary-70 border-t-4 border-t-primary-30 p-6 group hover:shadow-primary-30/10 transition-all duration-300 hover:-translate-y-1">
            <div class="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <i class="fa-solid fa-users text-8xl text-primary-30"></i>
            </div>

            <div class="mb-4">
                <div class="inline-flex p-3 bg-primary-70 rounded-xl text-primary-30 border border-primary-60">
                    <i class="fa-solid fa-user text-2xl"></i>
                </div>
            </div>

            <h3 class="font-primary text-xl font-bold text-white mb-1">User</h3>
            <p class="text-sm text-primary-20 mb-6 font-secondaryAndButton h-10">Standard access to stream music, create playlists, and manage profile.</p>

            <div class="border-t border-primary-70 pt-4">
                <div>
                    <p class="text-xs text-primary-20 font-bold uppercase tracking-wider">Total Accounts</p>
                    <p class="font-primary font-bold text-white text-2xl mt-1">{{ $role0 ?? 0 }}</p>
                </div>
            </div>
        </div>
    </div>

    {{-- ACCESS MATRIX --}}
    <div class="bg-primary-85 rounded-2xl shadow-lg border border-primary-70 overflow-hidden">
        {{-- Header --}}
        <div class="p-6 border-b border-primary-70 bg-primary-85/50 backdrop-blur-sm">
            <h3 class="font-primary text-lg font-bold text-white">Access Matrix</h3>
            <p class="text-sm text-primary-20">Detailed breakdown of permissions per role</p>
        </div>

        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="bg-primary-100/50 text-white text-sm font-bold uppercase tracking-wider border-b border-primary-70">
                        <th class="px-6 py-4 w-1/3">Feature / Module</th>
                        <th class="px-6 py-4 text-center text-accent-100">Superadmin</th>
                        <th class="px-6 py-4 text-center text-secondary-happy-100">Admin</th>
                        <th class="px-6 py-4 text-center text-primary-30">User</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-primary-70 text-sm font-secondaryAndButton">

                    {{-- SECTION 1 --}}
                    <tr class="bg-primary-70/30">
                        <td colspan="4" class="px-6 py-2 text-xs font-bold text-shadedOfGray-30 uppercase tracking-widest pl-6 border-l-4 border-primary-30">User Management</td>
                    </tr>
                    <tr class="hover:bg-primary-70/20 transition group">
                        <td class="px-6 py-4 font-medium text-shadedOfGray-20 group-hover:text-white transition-colors">View All Users</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                    </tr>
                    <tr class="hover:bg-primary-70/20 transition group">
                        <td class="px-6 py-4 font-medium text-shadedOfGray-20 group-hover:text-white transition-colors">Create/Edit Admins</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                    </tr>
                    <tr class="hover:bg-primary-70/20 transition group">
                        <td class="px-6 py-4 font-medium text-shadedOfGray-20 group-hover:text-white transition-colors">Delete Users</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                    </tr>

                    {{-- SECTION 2 --}}
                    <tr class="bg-primary-70/30">
                        <td colspan="4" class="px-6 py-2 text-xs font-bold text-shadedOfGray-30 uppercase tracking-widest pl-6 border-l-4 border-secondary-happy-100">Song & Content</td>
                    </tr>
                    <tr class="hover:bg-primary-70/20 transition group">
                        <td class="px-6 py-4 font-medium text-shadedOfGray-20 group-hover:text-white transition-colors">Upload Songs</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                    </tr>
                    <tr class="hover:bg-primary-70/20 transition group">
                        <td class="px-6 py-4 font-medium text-shadedOfGray-20 group-hover:text-white transition-colors">Edit/Delete Songs</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                    </tr>

                    <tr class="hover:bg-primary-70/20 transition group">
                        <td class="px-6 py-4 font-medium text-shadedOfGray-20 group-hover:text-white transition-colors">Create/Edit Playlists</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                    </tr>

                    {{-- SECTION 3 --}}
                    <tr class="bg-primary-70/30">
                        <td colspan="4" class="px-6 py-2 text-xs font-bold text-shadedOfGray-30 uppercase tracking-widest pl-6 border-l-4 border-primary-30">Access & Streaming</td>
                    </tr>
                    <tr class="hover:bg-primary-70/20 transition group">
                        <td class="px-6 py-4 font-medium text-shadedOfGray-20 group-hover:text-white transition-colors">Stream Music</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-primary-60 text-lg opacity-50"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-6 bg-primary-70/20 border-t border-primary-70">
            <p class="text-xs text-primary-20 italic flex items-center gap-2">
                <i class="fa-solid fa-circle-info text-primary-30"></i>
                Permissions are updated globally. Changes affect all users assigned to the role immediately.
            </p>
        </div>
    </div>

@endsection
