@extends('layouts.superadmin.master')

@section('title', 'Roles & Access')
@section('page_title', 'Roles & Permissions')
@section('page_subtitle', 'Manage user roles and system access levels')

@section('content')

    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">

        <div class="bg-white rounded-2xl shadow-lg border-t-4 border-accent-100 p-6 relative overflow-hidden group">
            <div class="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <i class="fa-solid fa-crown text-8xl text-accent-100"></i>
            </div>

            <div class="flex justify-between items-start mb-4">
                <div class="p-3 bg-accent-20 rounded-xl text-accent-100">
                    <i class="fa-solid fa-shield-halved text-2xl"></i>
                </div>
                <div class="flex -space-x-2">
                    <img class="w-8 h-8 rounded-full border-2 border-white" src="https://ui-avatars.com/api/?name=Super+Admin&background=E650C5&color=fff" alt="">
                    <img class="w-8 h-8 rounded-full border-2 border-white" src="https://ui-avatars.com/api/?name=Dev+Team&background=random" alt="">
                    <span class="flex items-center justify-center w-8 h-8 rounded-full border-2 border-white bg-shadedOfGray-10 text-xs font-bold text-shadedOfGray-60">+3</span>
                </div>
            </div>

            <h3 class="font-primary text-xl font-bold text-primary-100 mb-1">Superadmin</h3>
            <p class="text-sm text-shadedOfGray-60 mb-6 font-secondaryAndButton">Full access to all system features and settings.</p>

            <div class="flex justify-between items-center border-t border-shadedOfGray-10 pt-4">
                <div>
                    <p class="text-xs text-shadedOfGray-40">Total Accounts</p>
                    <p class="font-bold text-primary-100 text-lg">5</p>
                </div>
                <button class="text-accent-100 hover:text-accent-85 text-sm font-bold">Edit Role</button>
            </div>
        </div>

        <div class="bg-white rounded-2xl shadow-sm border-t-4 border-secondary-happy-100 p-6 relative overflow-hidden group">
            <div class="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <i class="fa-solid fa-user-gear text-8xl text-secondary-happy-100"></i>
            </div>

            <div class="flex justify-between items-start mb-4">
                <div class="p-3 bg-secondary-happy-20 rounded-xl text-secondary-happy-100">
                    <i class="fa-solid fa-laptop-file text-2xl"></i>
                </div>
                <div class="flex -space-x-2">
                    <img class="w-8 h-8 rounded-full border-2 border-white" src="https://ui-avatars.com/api/?name=Admin+One&background=FF8E2B&color=fff" alt="">
                    <img class="w-8 h-8 rounded-full border-2 border-white" src="https://ui-avatars.com/api/?name=Admin+Two&background=random" alt="">
                    <span class="flex items-center justify-center w-8 h-8 rounded-full border-2 border-white bg-shadedOfGray-10 text-xs font-bold text-shadedOfGray-60">+12</span>
                </div>
            </div>

            <h3 class="font-primary text-xl font-bold text-primary-100 mb-1">Admin</h3>
            <p class="text-sm text-shadedOfGray-60 mb-6 font-secondaryAndButton">Manage content, users, and day-to-day operations.</p>

            <div class="flex justify-between items-center border-t border-shadedOfGray-10 pt-4">
                <div>
                    <p class="text-xs text-shadedOfGray-40">Total Accounts</p>
                    <p class="font-bold text-primary-100 text-lg">14</p>
                </div>
                <button class="text-secondary-happy-100 hover:text-secondary-happy-85 text-sm font-bold">Edit Role</button>
            </div>
        </div>

        <div class="bg-white rounded-2xl shadow-sm border-t-4 border-primary-60 p-6 relative overflow-hidden group">
            <div class="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <i class="fa-solid fa-users text-8xl text-primary-60"></i>
            </div>

            <div class="flex justify-between items-start mb-4">
                <div class="p-3 bg-primary-10/50 rounded-xl text-primary-60">
                    <i class="fa-solid fa-user text-2xl"></i>
                </div>
                <div class="flex -space-x-2">
                    <img class="w-8 h-8 rounded-full border-2 border-white" src="https://ui-avatars.com/api/?name=User+A&background=142C80&color=fff" alt="">
                    <img class="w-8 h-8 rounded-full border-2 border-white" src="https://ui-avatars.com/api/?name=User+B&background=random" alt="">
                    <span class="flex items-center justify-center w-8 h-8 rounded-full border-2 border-white bg-shadedOfGray-10 text-xs font-bold text-shadedOfGray-60">+1k</span>
                </div>
            </div>

            <h3 class="font-primary text-xl font-bold text-primary-100 mb-1">User</h3>
            <p class="text-sm text-shadedOfGray-60 mb-6 font-secondaryAndButton">Standard access to stream music and manage profile.</p>

            <div class="flex justify-between items-center border-t border-shadedOfGray-10 pt-4">
                <div>
                    <p class="text-xs text-shadedOfGray-40">Total Accounts</p>
                    <p class="font-bold text-primary-100 text-lg">1,205</p>
                </div>
                <button class="text-primary-60 hover:text-primary-85 text-sm font-bold">View Details</button>
            </div>
        </div>
    </div>

    <div class="bg-white rounded-2xl shadow-lg border border-shadedOfGray-10 overflow-hidden">
        <div class="p-6 border-b border-shadedOfGray-10 flex flex-col md:flex-row justify-between items-center gap-4">
            <div>
                <h3 class="font-primary text-lg font-bold text-primary-100">Access Matrix</h3>
                <p class="text-sm text-shadedOfGray-60">Detailed breakdown of permissions per role</p>
            </div>
            <button class="bg-primary-100 text-white px-5 py-2.5 rounded-xl text-sm font-medium hover:bg-primary-85 transition-colors shadow-lg shadow-primary-30/30 flex items-center gap-2">
                <i class="fa-solid fa-plus"></i> Add New Permission
            </button>
        </div>

        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="bg-primary-10/30 text-primary-100 text-sm font-bold uppercase tracking-wider">
                        <th class="px-6 py-4 w-1/3">Feature / Module</th>
                        <th class="px-6 py-4 text-center text-accent-100">Superadmin</th>
                        <th class="px-6 py-4 text-center text-secondary-happy-100">Admin</th>
                        <th class="px-6 py-4 text-center text-primary-60">User</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-shadedOfGray-10 text-sm font-secondaryAndButton">

                    <tr class="bg-shadedOfGray-10/30">
                        <td colspan="4" class="px-6 py-2 text-xs font-bold text-shadedOfGray-60 uppercase tracking-widest">User Management</td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4 font-medium text-primary-85">View All Users</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-shadedOfGray-20 text-lg"></i></td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4 font-medium text-primary-85">Create/Edit Admins</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-shadedOfGray-20 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-shadedOfGray-20 text-lg"></i></td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4 font-medium text-primary-85">Ban/Delete Users</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-shadedOfGray-20 text-lg"></i></td>
                    </tr>

                    <tr class="bg-shadedOfGray-10/30">
                        <td colspan="4" class="px-6 py-2 text-xs font-bold text-shadedOfGray-60 uppercase tracking-widest">Song & Content</td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4 font-medium text-primary-85">Upload Songs</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-shadedOfGray-20 text-lg"></i></td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4 font-medium text-primary-85">Stream Music</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4 font-medium text-primary-85">Create Playlists</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                    </tr>

                    <tr class="bg-shadedOfGray-10/30">
                        <td colspan="4" class="px-6 py-2 text-xs font-bold text-shadedOfGray-60 uppercase tracking-widest">System Settings</td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4 font-medium text-primary-85">Manage Roles</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-shadedOfGray-20 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-shadedOfGray-20 text-lg"></i></td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4 font-medium text-primary-85">View Audit Logs</td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-check text-secondary-relaxed-100 text-lg"></i></td>
                        <td class="px-6 py-4 text-center"><i class="fa-solid fa-circle-xmark text-shadedOfGray-20 text-lg"></i></td>
                    </tr>

                </tbody>
            </table>
        </div>

        <div class="p-6 bg-primary-10/10 border-t border-shadedOfGray-10">
            <p class="text-xs text-shadedOfGray-60 italic">
                <i class="fa-solid fa-circle-info mr-1 text-primary-30"></i>
                Permissions are updated globally. Changes affect all users assigned to the role immediately.
            </p>
        </div>
    </div>

@endsection
