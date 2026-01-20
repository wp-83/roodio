@extends('layouts.superadmin.master')

@section('title', 'User Management')
@section('page_title', 'User Management')
@section('page_subtitle', 'Manage system users permissions')

@section('content')
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6 mb-8">
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center justify-between">
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1">Total Users</p>
                <h3 class="font-primary text-2xl lg:text-3xl text-primary-100 font-bold">{{ $users->count() }}</h3>
            </div>
            <div class="w-12 h-12 lg:w-14 lg:h-14 rounded-xl bg-secondary-happy-20 flex items-center justify-center text-secondary-happy-100 text-xl lg:text-2xl">
                <i class="fa-solid fa-users"></i>
            </div>
        </div>

        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center justify-between">
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1">Active Users</p>
                <h3 class="font-primary text-2xl lg:text-3xl text-primary-100 font-bold">890</h3>
            </div>
            <div class="w-12 h-12 lg:w-14 lg:h-14 rounded-xl bg-secondary-relaxed-20 flex items-center justify-center text-secondary-relaxed-100 text-xl lg:text-2xl">
                <i class="fa-solid fa-user-check"></i>
            </div>
        </div>

        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center justify-between sm:col-span-2 lg:col-span-1">
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1">New Registrations</p>
                <h3 class="font-primary text-2xl lg:text-3xl text-primary-100 font-bold">{{ $totalNewUser }}</h3>
            </div>
            <div class="w-12 h-12 lg:w-14 lg:h-14 rounded-xl bg-accent-20 flex items-center justify-center text-accent-100 text-xl lg:text-2xl">
                <i class="fa-solid fa-user-plus"></i>
            </div>
        </div>
    </div>

    <div class="bg-white rounded-2xl shadow-lg border border-shadedOfGray-10 overflow-hidden">

        <div class="p-4 lg:p-6 border-b border-shadedOfGray-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div class="relative w-full md:w-80 lg:w-96">
                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-shadedOfGray-40">
                    <i class="fa-solid fa-magnifying-glass"></i>
                </span>
                <input type="text" placeholder="Search users..." class="w-full pl-10 pr-4 py-2.5 lg:py-3 rounded-xl border border-shadedOfGray-20 bg-primary-10/30 text-primary-100 focus:outline-none focus:border-accent-100 focus:ring-1 focus:ring-accent-100 text-sm transition-all">
            </div>

            <div class="flex gap-2 lg:gap-3 w-full md:w-auto">
                <button class="flex-1 md:flex-none px-4 lg:px-5 py-2.5 lg:py-3 rounded-xl border border-shadedOfGray-20 text-primary-70 font-medium hover:bg-shadedOfGray-10 transition-colors flex items-center justify-center gap-2 text-sm">
                    <i class="fa-solid fa-filter"></i> <span class="hidden sm:inline">Filter</span>
                </button>
                <button class="flex-1 md:flex-none bg-accent-100 hover:bg-accent-85 text-white px-4 lg:px-6 py-2.5 lg:py-3 rounded-xl font-medium shadow-lg shadow-accent-50/50 transition-all flex items-center justify-center gap-2 text-sm">
                    <i class="fa-solid fa-plus"></i> <span class="whitespace-nowrap">Add User</span>
                </button>
            </div>
        </div>

        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse min-w-[800px]">
                <thead class="bg-primary-100 text-white">
                    <tr>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide">User</th>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide">Role</th>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide">Status</th>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide">Joined Date</th>
                        <th class="px-6 py-4 text-sm font-semibold tracking-wide text-right">Actions</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-shadedOfGray-10">
                    <tr class="hover:bg-primary-10/30 transition-colors duration-150 group">
                        <td class="px-6 py-4">
                            <div class="flex items-center gap-3">
                                <img src="https://ui-avatars.com/api/?name=Budi+Santoso&background=random" class="w-10 h-10 rounded-full object-cover">
                                <div>
                                    <p class="text-sm lg:text-base font-bold text-primary-100 group-hover:text-accent-100 transition-colors">Budi Santoso</p>
                                    <p class="text-xs text-shadedOfGray-60">budi@example.com</p>
                                </div>
                            </div>
                        </td>
                        <td class="px-6 py-4">
                            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-primary-20 text-primary-100">Editor</span>
                        </td>
                        <td class="px-6 py-4">
                            <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold bg-secondary-relaxed-20 text-secondary-relaxed-100 border border-secondary-relaxed-30">
                                <span class="w-1.5 h-1.5 rounded-full bg-secondary-relaxed-100"></span> Active
                            </span>
                        </td>
                        <td class="px-6 py-4 text-sm text-shadedOfGray-70">12 Jan 2024</td>
                        <td class="px-6 py-4 text-right">
                            <div class="flex items-center justify-end gap-2">
                                <button class="w-8 h-8 rounded-lg flex items-center justify-center text-accent-100 hover:bg-accent-20 transition-colors"><i class="fa-solid fa-pen-to-square"></i></button>
                                <button class="w-8 h-8 rounded-lg flex items-center justify-center text-secondary-angry-100 hover:bg-secondary-angry-20 transition-colors"><i class="fa-solid fa-trash-can"></i></button>
                            </div>
                        </td>
                    </tr>
                    </tbody>
            </table>
        </div>

        <div class="p-4 lg:p-6 border-t border-shadedOfGray-10 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p class="text-sm text-shadedOfGray-60 text-center sm:text-left">
                Showing <span class="font-bold text-primary-100">1</span> to <span class="font-bold text-primary-100">3</span> of <span class="font-bold text-primary-100">50</span> users
            </p>
            <div class="flex gap-2">
                <button class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-60 hover:bg-primary-10 text-sm">Prev</button>
                <button class="px-3 py-1.5 rounded-lg bg-primary-100 text-white hover:bg-primary-85 text-sm">1</button>
                <button class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-60 hover:bg-primary-10 text-sm">2</button>
                <button class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-60 hover:bg-primary-10 text-sm">Next</button>
            </div>
        </div>
    </div>
@endsection
