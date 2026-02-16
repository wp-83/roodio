@extends('layouts.superadmin.master')

@section('title', 'User Overview')
@section('page_title', 'User Overview')
@section('page_subtitle', 'Monitoring user statistics and demographics')

@section('content')
    {{-- Load Chart.js UMD --}}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

    {{-- 1. STATS CARDS GRID --}}
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">

        {{-- Card 1: Total Users --}}
        <div class="relative overflow-hidden bg-gradient-to-br from-primary-85 to-primary-100 rounded-2xl p-6 border border-primary-70 shadow-lg group hover:border-primary-50 transition-all duration-300">
            <div class="flex items-center gap-4 relative z-10">
                <div class="p-4 bg-primary-70 rounded-xl text-white group-hover:bg-white group-hover:text-primary-100 transition-colors duration-300 shadow-md">
                    <i class="fa-solid fa-users text-2xl"></i>
                </div>
                <div>
                    <p class="text-sm text-[#9CA3AF] mb-1 font-secondaryAndButton font-bold uppercase tracking-wider">Total Users</p>
                    <h3 class="font-primary text-2xl font-bold text-white">{{ number_format($totalUser) }}</h3>
                    <p class="text-xs text-shadedOfGray-30 mt-1">Registered accounts</p>
                </div>
            </div>
            {{-- Decorative Blob --}}
            <div class="absolute -bottom-6 -right-6 w-24 h-24 bg-white/5 rounded-full blur-2xl group-hover:bg-white/10 transition-colors"></div>
        </div>

        {{-- Card 2: New Users Today --}}
        <div class="relative overflow-hidden bg-gradient-to-br from-primary-85 to-primary-100 rounded-2xl p-6 border border-primary-70 shadow-lg group hover:border-secondary-relaxed-100 transition-all duration-300">
            <div class="flex items-center gap-4 relative z-10">
                <div class="p-4 bg-primary-70 rounded-xl text-secondary-relaxed-100 group-hover:bg-secondary-relaxed-100 group-hover:text-white transition-colors duration-300 shadow-md">
                    <i class="fa-solid fa-user-plus text-2xl"></i>
                </div>
                <div>
                    <p class="text-sm text-[#9CA3AF] mb-1 font-secondaryAndButton font-bold uppercase tracking-wider">New Users</p>
                    <h3 class="font-primary text-2xl font-bold text-white">{{ number_format($newUsersToday) }}</h3>
                    <p class="text-xs text-secondary-relaxed-100 font-bold mt-1 flex items-center gap-1">
                        <i class="fa-solid fa-calendar-day"></i> Joined Today
                    </p>
                </div>
            </div>
            <div class="absolute -bottom-6 -right-6 w-24 h-24 bg-secondary-relaxed-100/10 rounded-full blur-2xl group-hover:bg-secondary-relaxed-100/20 transition-colors"></div>
        </div>

        {{-- Card 3: Administrators --}}
        <div class="relative overflow-hidden bg-gradient-to-br from-primary-85 to-primary-100 rounded-2xl p-6 border border-primary-70 shadow-lg group hover:border-secondary-happy-100 transition-all duration-300">
            <div class="flex items-center gap-4 relative z-10">
                <div class="p-4 bg-primary-70 rounded-xl text-secondary-happy-100 group-hover:bg-secondary-happy-100 group-hover:text-white transition-colors duration-300 shadow-md">
                    <i class="fa-solid fa-user-shield text-2xl"></i>
                </div>
                <div>
                    <p class="text-sm text-[#9CA3AF] mb-1 font-secondaryAndButton font-bold uppercase tracking-wider">Admins</p>
                    <h3 class="font-primary text-2xl font-bold text-white">{{ number_format($totalAdmins) }}</h3>
                    <p class="text-xs text-shadedOfGray-30 mt-1">Admin & Superadmin</p>
                </div>
            </div>
            <div class="absolute -bottom-6 -right-6 w-24 h-24 bg-secondary-happy-100/10 rounded-full blur-2xl group-hover:bg-secondary-happy-100/20 transition-colors"></div>
        </div>

        {{-- Card 4: Demographics --}}
        <div class="relative overflow-hidden bg-gradient-to-br from-primary-85 to-primary-100 rounded-2xl p-6 border border-primary-70 shadow-lg group hover:border-accent-100 transition-all duration-300">
            <div class="flex items-center gap-3 relative z-10">
                {{-- Icon Wrapper: Tambah flex-shrink-0 agar icon tidak gepeng --}}
                <div class="p-4 bg-primary-70 rounded-xl text-accent-100 group-hover:bg-accent-100 group-hover:text-white transition-colors duration-300 shadow-md flex-shrink-0">
                    <i class="fa-solid fa-venus-mars text-2xl"></i>
                </div>

                {{-- Text Wrapper: Tambah flex-1 dan min-w-0 agar teks menyesuaikan ruang --}}
                <div class="flex-1 min-w-0">
                    {{-- Title: Tambah truncate --}}
                    <p class="text-sm text-[#9CA3AF] mb-1 font-secondaryAndButton font-bold uppercase tracking-wider truncate" title="Demographics">
                        Demographics
                    </p>

                    <div class="flex items-baseline gap-2">
                        <h3 class="font-primary text-xl font-bold text-white">{{ $totalMale }} <span class="text-xs font-normal text-[#9CA3AF]">M</span></h3>
                        <span class="text-shadedOfGray-60">/</span>
                        <h3 class="font-primary text-xl font-bold text-white">{{ $totalFemale }} <span class="text-xs font-normal text-[#9CA3AF]">F</span></h3>
                    </div>
                    <p class="text-xs text-shadedOfGray-30 mt-1 truncate">Male vs Female</p>
                </div>
            </div>
            <div class="absolute -bottom-6 -right-6 w-24 h-24 bg-accent-100/10 rounded-full blur-2xl group-hover:bg-accent-100/20 transition-colors"></div>
        </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">

        {{-- 2. USER DISTRIBUTION CHART (1/3 Width) --}}
        <div class="bg-primary-85 rounded-2xl border border-primary-70 shadow-lg flex flex-col justify-between lg:col-span-1 h-full">
            <div class="p-6 border-b border-primary-70">
                <h3 class="font-primary text-lg font-bold text-white mb-1">User Distribution</h3>
                <p class="text-xs text-[#9CA3AF]">By Role / Access Level</p>
            </div>

            {{-- Chart Container --}}
            <div class="relative w-full flex justify-center py-6 px-4" style="height: 240px;">
                <canvas id="roleChart"></canvas>
            </div>

            {{-- Legend / Stats --}}
            <div class="p-6 pt-0 space-y-4">
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-3 h-3 rounded-full bg-primary-30"></span>
                        <span class="text-shadedOfGray-30">Users</span>
                    </div>
                    <span class="font-bold text-white">{{ $perc0 }}%</span>
                </div>
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-3 h-3 rounded-full bg-accent-100"></span>
                        <span class="text-shadedOfGray-30">Admins</span>
                    </div>
                    <span class="font-bold text-white">{{ $perc1 }}%</span>
                </div>
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-3 h-3 rounded-full bg-secondary-happy-100"></span>
                        <span class="text-shadedOfGray-30">Super Admins</span>
                    </div>
                    <span class="font-bold text-white">{{ $perc2 }}%</span>
                </div>
            </div>
        </div>

        {{-- 3. LATEST REGISTRATIONS TABLE (2/3 Width) --}}
        <div class="bg-primary-85 rounded-2xl border border-primary-70 shadow-lg overflow-hidden flex flex-col lg:col-span-2 h-full">
            <div class="p-6 border-b border-primary-70 flex justify-between items-center bg-primary-85/50 backdrop-blur-sm">
                <div>
                    <h3 class="font-primary text-lg font-bold text-white">Latest Registrations</h3>
                    <p class="text-xs text-[#9CA3AF]">Recently joined users</p>
                </div>
                <a href="{{ route('superadmin.users.index') }}" class="text-xs font-bold text-secondary-happy-100 hover:text-white transition-colors flex items-center gap-1">
                    VIEW ALL <i class="fa-solid fa-arrow-right text-[10px]"></i>
                </a>
            </div>

            <div class="overflow-x-auto flex-grow">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="text-xs text-[#9CA3AF] font-secondaryAndButton border-b border-primary-70 bg-primary-85/30 uppercase tracking-wider">
                            <th class="px-6 py-4 font-bold">User</th>
                            <th class="px-6 py-4 font-bold">Role</th>
                            <th class="px-6 py-4 font-bold">Gender</th>
                            <th class="px-6 py-4 font-bold text-right">Date</th>
                        </tr>
                    </thead>
                    <tbody class="text-sm divide-y divide-primary-70">
                        @foreach ($users as $user)
                        <tr class="group hover:bg-primary-70/30 transition-colors">
                            <td class="px-6 py-4">
                                <div class="flex items-center gap-3">
                                    {{-- Avatar --}}
                                    <div class="flex-shrink-0">
                                        @if($user->userDetail?->profilePhoto)
                                            <img src="{{ config('filesystems.disks.azure.url') . '/' . $user->userDetail->profilePhoto }}" class="w-9 h-9 rounded-full object-cover border border-primary-60">
                                        @else
                                            <div class="w-9 h-9 rounded-full bg-primary-60 flex items-center justify-center text-white font-bold text-xs border border-primary-50">
                                                {{ substr($user->username, 0, 1) }}
                                            </div>
                                        @endif
                                    </div>
                                    {{-- Name & Email --}}
                                    <div>
                                        <p class="font-bold text-white group-hover:text-secondary-happy-100 transition-colors">{{ $user->userDetail?->fullname ?? $user->username }}</p>
                                        <p class="text-xs text-[#9CA3AF]">{{ $user->userDetail?->email ?? $user->email }}</p>
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
                            <td class="px-6 py-4 text-shadedOfGray-30 text-xs">
                                @if($user->userDetail?->gender === 1)
                                    <span class="flex items-center gap-1"><i class="fa-solid fa-mars text-blue-400"></i> Male</span>
                                @elseif($user->userDetail?->gender === 0)
                                    <span class="flex items-center gap-1"><i class="fa-solid fa-venus text-pink-400"></i> Female</span>
                                @else
                                    <span class="text-shadedOfGray-60">-</span>
                                @endif
                            </td>
                            <td class="px-6 py-4 text-right text-[#9CA3AF] text-xs font-mono">
                                {{ $user->created_at->format('d M Y') }}
                            </td>
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
                const role0 = {{ $role0 }}; // User
                const role1 = {{ $role1 }}; // Admin
                const role2 = {{ $role2 }}; // Superadmin
                const totalRoles = role0 + role1 + role2;

                let chartData = [role0, role1, role2];
                // Warna Chart disesuaikan dengan tema Dark Mode
                // Users: Primary-30 (Biru Terang), Admins: Accent (Pink), Superadmin: Happy (Orange)
                let chartColors = ['#4F6CC3', '#E650C5', '#FF8E2B'];
                let chartLabels = ['Users', 'Admins', 'Super Admins'];

                if (totalRoles === 0) {
                    chartData = [1];
                    chartColors = ['#1F3A98']; // Primary-50 (Darker Blue) for Empty
                    chartLabels = ['No Data'];
                }

                new Chart(ctxRole.getContext('2d'), {
                    type: 'doughnut',
                    data: {
                        labels: chartLabels,
                        datasets: [{
                            data: chartData,
                            backgroundColor: chartColors,
                            borderWidth: 0, // No border for cleaner look
                            hoverOffset: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '75%', // Lebih tipis agar modern
                        plugins: {
                            legend: { display: false }, // Legend custom di HTML
                            tooltip: {
                                enabled: totalRoles !== 0,
                                backgroundColor: '#020A36', // Primary-100
                                titleColor: '#FFFFFF',
                                bodyColor: '#B2B2B2', // Shaded-30
                                borderColor: '#0D1F67', // Primary-70
                                borderWidth: 1,
                                padding: 10,
                                cornerRadius: 8,
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
