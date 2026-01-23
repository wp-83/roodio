@extends('layouts.superadmin.master')

@section('title', 'User Overview')
@section('page_title', 'User Overview')
@section('page_subtitle', 'Monitoring user statistics and demographics')

@section('content')
    {{-- Load Chart.js UMD --}}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

    {{-- STATS CARDS GRID --}}
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 mb-8">
        {{-- Card 1: Total Users --}}
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center gap-4 group hover:shadow-md transition-shadow">
            <div class="p-4 bg-primary-10/30 rounded-xl text-primary-100 group-hover:bg-primary-100 group-hover:text-white transition-colors duration-300">
                <i class="fa-solid fa-users text-2xl"></i>
            </div>
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1 font-secondaryAndButton">Total Users</p>
                <h3 class="font-primary text-2xl font-bold text-primary-100">{{ $totalUser }}</h3>
                <p class="text-xs text-secondary-relaxed-100 font-bold mt-1"><span class="text-shadedOfGray-40 font-normal">Registered accounts</span></p>
            </div>
        </div>

        {{-- Card 2: New Users Today --}}
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center gap-4 group hover:shadow-md transition-shadow">
            <div class="p-4 bg-secondary-relaxed-20 rounded-xl text-secondary-relaxed-100 group-hover:bg-secondary-relaxed-100 group-hover:text-white transition-colors duration-300">
                <i class="fa-solid fa-user-plus text-2xl"></i>
            </div>
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1 font-secondaryAndButton">New Users Today</p>
                <h3 class="font-primary text-2xl font-bold text-primary-100">{{ $newUsersToday }}</h3>
                <p class="text-xs text-secondary-relaxed-100 font-bold mt-1"><i class="fa-solid fa-calendar-day"></i> <span class="text-shadedOfGray-40 font-normal">Joined today</span></p>
            </div>
        </div>

        {{-- Card 3: Administrators --}}
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center gap-4 group hover:shadow-md transition-shadow">
            <div class="p-4 bg-secondary-happy-20 rounded-xl text-secondary-happy-100 group-hover:bg-secondary-happy-100 group-hover:text-white transition-colors duration-300">
                <i class="fa-solid fa-user-shield text-2xl"></i>
            </div>
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1 font-secondaryAndButton">Administrators</p>
                <h3 class="font-primary text-2xl font-bold text-primary-100">{{ $totalAdmins }}</h3>
                <p class="text-xs text-shadedOfGray-40 mt-1">Admin & Superadmin</p>
            </div>
        </div>

        {{-- Card 4: Demographics --}}
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center gap-4 group hover:shadow-md transition-shadow">
            <div class="p-4 bg-accent-20 rounded-xl text-accent-100 group-hover:bg-accent-100 group-hover:text-white transition-colors duration-300">
                <i class="fa-solid fa-venus-mars text-2xl"></i>
            </div>
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1 font-secondaryAndButton">Demographics</p>
                <div class="flex items-baseline gap-2">
                    <h3 class="font-primary text-xl font-bold text-primary-100">{{ $totalMale }} <span class="text-xs font-normal text-shadedOfGray-60">M</span></h3>
                    <span class="text-shadedOfGray-40">/</span>
                    <h3 class="font-primary text-xl font-bold text-primary-100">{{ $totalFemale }} <span class="text-xs font-normal text-shadedOfGray-60">F</span></h3>
                </div>
                <p class="text-xs text-shadedOfGray-40 mt-1">Male vs Female</p>
            </div>
        </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {{-- USER DISTRIBUTION CHART (Full Width on Mobile, 1/3 on Desktop if alongside something else, or make it wider) --}}
        {{-- Since we removed the trend chart, we can center this or make it smaller --}}
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex flex-col justify-between lg:col-span-1">
            <div>
                <h3 class="font-primary text-lg font-bold text-primary-100 mb-1">User Distribution</h3>
                <p class="text-xs text-shadedOfGray-60 mb-4">By Role / Access Level</p>
            </div>

            {{-- Chart Container --}}
            <div class="relative w-full flex justify-center mb-4" style="height: 220px; min-height: 220px;">
                <canvas id="roleChart"></canvas>
            </div>

            {{-- Legend / Stats --}}
            <div class="space-y-3">
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-primary-85"></span>
                        <span class="text-shadedOfGray-60">Users</span>
                    </div>
                    <span class="font-bold text-primary-100">{{ $perc0 }}%</span>
                </div>
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-accent-100"></span>
                        <span class="text-shadedOfGray-60">Admins</span>
                    </div>
                    <span class="font-bold text-primary-100">{{ $perc1 }}%</span>
                </div>
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-secondary-happy-100"></span>
                        <span class="text-shadedOfGray-60">Super Admins</span>
                    </div>
                    <span class="font-bold text-primary-100">{{ $perc2 }}%</span>
                </div>
            </div>
        </div>

        {{-- LATEST REGISTRATIONS TABLE (Takes up remaining space) --}}
        <div class="bg-white rounded-2xl shadow-sm border border-shadedOfGray-10 overflow-hidden lg:col-span-2">
            <div class="p-6 border-b border-shadedOfGray-10 flex justify-between items-center">
                <div>
                    <h3 class="font-primary text-lg font-bold text-primary-100">Latest Registrations</h3>
                    <p class="text-xs text-shadedOfGray-60">Recently joined users</p>
                </div>
                <a href="{{ route('superadmin.users.index') }}" class="text-xs font-bold text-accent-100 hover:text-accent-85 flex items-center gap-1">
                    View All <i class="fa-solid fa-arrow-right"></i>
                </a>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left">
                    <thead class="bg-primary-10/30 text-primary-85 text-xs font-bold uppercase tracking-wider">
                        <tr>
                            <th class="px-6 py-4">User</th>
                            <th class="px-6 py-4">Role</th>
                            <th class="px-6 py-4">Gender</th>
                            <th class="px-6 py-4 text-right">Date</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-shadedOfGray-10 text-sm">
                        @foreach ($users as $user)
                        <tr class="hover:bg-primary-10/10 transition">
                            <td class="px-6 py-4">
                                <div class="flex items-center gap-3">
                                    @if($user->userDetail?->profilePhoto)
                                        <img src="{{ config('filesystems.disks.azure.url') . '/' . $user->userDetail->profilePhoto }}" class="w-8 h-8 rounded-full object-cover">
                                    @else
                                        <div class="w-8 h-8 rounded-full bg-accent-20 flex items-center justify-center text-accent-100 font-bold text-xs">
                                            {{ substr($user->username, 0, 1) }}
                                        </div>
                                    @endif
                                    <div>
                                        <p class="font-bold text-primary-100">{{ $user->userDetail?->fullname ?? $user->username }}</p>
                                        <p class="text-xs text-shadedOfGray-60">{{ $user->userDetail?->email ?? $user->email }}</p>
                                    </div>
                                </div>
                            </td>
                            <td class="px-6 py-4">
                                @php
                                    $roleName = match($user->role) { 0 => 'User', 1 => 'Admin', 2 => 'Superadmin', default => 'Unknown' };
                                    $roleClass = match($user->role) { 0 => 'bg-primary-20 text-primary-100', 1 => 'bg-secondary-happy-20 text-secondary-happy-100', 2 => 'bg-accent-20 text-accent-100', default => 'bg-gray-200' };
                                @endphp
                                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold {{ $roleClass }}">{{ $roleName }}</span>
                            </td>
                            <td class="px-6 py-4 text-shadedOfGray-60">
                                {{ match($user->userDetail?->gender) { 1 => 'Male', 0 => 'Female', default => '-' } }}
                            </td>
                            <td class="px-6 py-4 text-right text-shadedOfGray-60">{{ $user->created_at->format('d M Y') }}</td>
                        </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    {{-- SCRIPTS --}}
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            // --- User Roles Chart (Doughnut) ---
            const ctxRole = document.getElementById('roleChart');

            if (ctxRole) {
                const role0 = {{ $role0 }};
                const role1 = {{ $role1 }};
                const role2 = {{ $role2 }};
                const totalRoles = role0 + role1 + role2;

                let chartData = [role0, role1, role2];
                let chartColors = ['#06134D', '#E650C5', '#FF8E2B']; // Primary, Accent, Happy
                let chartLabels = ['Users', 'Admins', 'Super Admins'];

                // Handle Empty State
                if (totalRoles === 0) {
                    chartData = [1];
                    chartColors = ['#E5E7EB']; // Gray
                    chartLabels = ['No Data'];
                }

                new Chart(ctxRole.getContext('2d'), {
                    type: 'doughnut',
                    data: {
                        labels: chartLabels,
                        datasets: [{
                            data: chartData,
                            backgroundColor: chartColors,
                            borderWidth: 0,
                            hoverOffset: totalRoles === 0 ? 0 : 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '75%',
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                enabled: totalRoles !== 0,
                                callbacks: {
                                    label: function(context) {
                                        let value = context.raw || 0;
                                        return ' ' + value + ' Accounts';
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
    </script>
@endsection
