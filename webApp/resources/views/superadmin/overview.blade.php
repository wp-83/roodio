@extends('layouts.superadmin.master')

@section('title', 'User Overview')
@section('page_title', 'User Overview')
@section('page_subtitle', 'Monitoring user statistics and registration trends')

@section('content')
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 mb-8">

        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center gap-4 relative overflow-hidden group hover:shadow-md transition-shadow">
            <div class="p-4 bg-primary-10/30 rounded-xl text-primary-100 group-hover:bg-primary-100 group-hover:text-white transition-colors duration-300">
                <i class="fa-solid fa-users text-2xl"></i>
            </div>
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1 font-secondaryAndButton">Total Users</p>
                <h3 class="font-primary text-2xl font-bold text-primary-100">1,240</h3>
                <p class="text-xs text-secondary-relaxed-100 font-bold flex items-center gap-1 mt-1">
                    <i class="fa-solid fa-arrow-trend-up"></i> +5% <span class="text-shadedOfGray-40 font-normal">this month</span>
                </p>
            </div>
        </div>

        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center gap-4 relative overflow-hidden group hover:shadow-md transition-shadow">
            <div class="p-4 bg-secondary-relaxed-20 rounded-xl text-secondary-relaxed-100 group-hover:bg-secondary-relaxed-100 group-hover:text-white transition-colors duration-300">
                <i class="fa-solid fa-user-check text-2xl"></i>
            </div>
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1 font-secondaryAndButton">Active Accounts</p>
                <h3 class="font-primary text-2xl font-bold text-primary-100">1,105</h3>
                <p class="text-xs text-shadedOfGray-40 mt-1">
                    <span class="font-bold text-primary-100">89%</span> retention rate
                </p>
            </div>
        </div>

        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center gap-4 relative overflow-hidden group hover:shadow-md transition-shadow">
            <div class="p-4 bg-secondary-happy-20 rounded-xl text-secondary-happy-100 group-hover:bg-secondary-happy-100 group-hover:text-white transition-colors duration-300">
                <i class="fa-solid fa-user-clock text-2xl"></i>
            </div>
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1 font-secondaryAndButton">Pending Review</p>
                <h3 class="font-primary text-2xl font-bold text-primary-100">45</h3>
                <p class="text-xs text-secondary-happy-100 font-bold mt-1">Needs Attention</p>
            </div>
        </div>

        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex items-center gap-4 relative overflow-hidden group hover:shadow-md transition-shadow">
            <div class="p-4 bg-secondary-angry-20 rounded-xl text-secondary-angry-100 group-hover:bg-secondary-angry-100 group-hover:text-white transition-colors duration-300">
                <i class="fa-solid fa-user-slash text-2xl"></i>
            </div>
            <div>
                <p class="text-sm text-shadedOfGray-60 mb-1 font-secondaryAndButton">Banned Users</p>
                <h3 class="font-primary text-2xl font-bold text-primary-100">90</h3>
                <p class="text-xs text-secondary-angry-100 font-bold flex items-center gap-1 mt-1">
                    <i class="fa-solid fa-arrow-trend-down"></i> +2 <span class="text-shadedOfGray-40 font-normal">this week</span>
                </p>
            </div>
        </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">

        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 lg:col-span-2">
            <div class="flex items-center justify-between mb-6">
                <div>
                    <h3 class="font-primary text-lg font-bold text-primary-100">User Registration Trend</h3>
                    <p class="text-xs text-shadedOfGray-60">New user signups over the last 6 months</p>
                </div>
                <button class="text-xs border border-shadedOfGray-20 rounded-lg px-3 py-1 text-shadedOfGray-60 hover:bg-primary-10 transition-colors">
                    <i class="fa-solid fa-download mr-1"></i> Report
                </button>
            </div>
            <div class="relative h-64 w-full">
                <canvas id="registrationChart"></canvas>
            </div>
        </div>

        <div class="bg-white p-6 rounded-2xl shadow-sm border border-shadedOfGray-10 flex flex-col justify-between">
            <div>
                <h3 class="font-primary text-lg font-bold text-primary-100 mb-1">User Distribution</h3>
                <p class="text-xs text-shadedOfGray-60 mb-4">By Role / Access Level</p>
            </div>

            <div class="relative h-48 w-full flex justify-center mb-4">
                <canvas id="roleChart"></canvas>
            </div>

            <div class="space-y-3">
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-primary-85"></span>
                        <span class="text-shadedOfGray-60">Regular Users</span>
                    </div>
                    <span class="font-bold text-primary-100">85%</span>
                </div>
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-accent-100"></span>
                        <span class="text-shadedOfGray-60">Editors/Staff</span>
                    </div>
                    <span class="font-bold text-primary-100">10%</span>
                </div>
                <div class="flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-secondary-happy-100"></span>
                        <span class="text-shadedOfGray-60">Super Admins</span>
                    </div>
                    <span class="font-bold text-primary-100">5%</span>
                </div>
            </div>
        </div>
    </div>

    <div class="bg-white rounded-2xl shadow-sm border border-shadedOfGray-10 overflow-hidden">
        <div class="p-6 border-b border-shadedOfGray-10 flex justify-between items-center">
            <div>
                <h3 class="font-primary text-lg font-bold text-primary-100">Latest Registrations</h3>
                <p class="text-xs text-shadedOfGray-60">Users who joined in the last 24 hours</p>
            </div>
            <a href="{{ route('superadmin.users.index') }}" class="text-xs font-bold text-accent-100 hover:text-accent-85 flex items-center gap-1">
                View All Users <i class="fa-solid fa-arrow-right"></i>
            </a>
        </div>

        <div class="overflow-x-auto">
            <table class="w-full text-left">
                <thead class="bg-primary-10/30 text-primary-85 text-xs font-bold uppercase tracking-wider">
                    <tr>
                        <th class="px-6 py-4">User Details</th>
                        <th class="px-6 py-4">Role</th>
                        <th class="px-6 py-4">Registered At</th>
                        <th class="px-6 py-4 text-right">Status</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-shadedOfGray-10 text-sm">
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4">
                            <div class="flex items-center gap-3">
                                <div class="w-8 h-8 rounded-full bg-accent-20 flex items-center justify-center text-accent-100 font-bold text-xs">
                                    AD
                                </div>
                                <div>
                                    <p class="font-bold text-primary-100">Andi Pratama</p>
                                    <p class="text-xs text-shadedOfGray-60">andi@gmail.com</p>
                                </div>
                            </div>
                        </td>
                        <td class="px-6 py-4 text-shadedOfGray-60">Regular User</td>
                        <td class="px-6 py-4 text-shadedOfGray-60">2 mins ago</td>
                        <td class="px-6 py-4 text-right">
                            <span class="px-2.5 py-1 bg-secondary-happy-20 text-secondary-happy-100 rounded-md text-xs font-bold">Pending</span>
                        </td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4">
                            <div class="flex items-center gap-3">
                                <img src="https://ui-avatars.com/api/?name=Sarah+Lee&background=random" class="w-8 h-8 rounded-full">
                                <div>
                                    <p class="font-bold text-primary-100">Sarah Lee</p>
                                    <p class="text-xs text-shadedOfGray-60">sarah.lee@corp.com</p>
                                </div>
                            </div>
                        </td>
                        <td class="px-6 py-4 text-shadedOfGray-60">Editor</td>
                        <td class="px-6 py-4 text-shadedOfGray-60">1 hour ago</td>
                        <td class="px-6 py-4 text-right">
                            <span class="px-2.5 py-1 bg-secondary-relaxed-20 text-secondary-relaxed-100 rounded-md text-xs font-bold">Active</span>
                        </td>
                    </tr>
                    <tr class="hover:bg-primary-10/10 transition">
                        <td class="px-6 py-4">
                            <div class="flex items-center gap-3">
                                <img src="https://ui-avatars.com/api/?name=Michael+Jordan&background=0D1F67&color=fff" class="w-8 h-8 rounded-full">
                                <div>
                                    <p class="font-bold text-primary-100">Michael J.</p>
                                    <p class="text-xs text-shadedOfGray-60">mike@web.id</p>
                                </div>
                            </div>
                        </td>
                        <td class="px-6 py-4 text-shadedOfGray-60">Regular User</td>
                        <td class="px-6 py-4 text-shadedOfGray-60">3 hours ago</td>
                        <td class="px-6 py-4 text-right">
                            <span class="px-2.5 py-1 bg-secondary-relaxed-20 text-secondary-relaxed-100 rounded-md text-xs font-bold">Active</span>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // --- 1. User Growth Chart (Line) ---
        const ctx1 = document.getElementById('registrationChart').getContext('2d');
        new Chart(ctx1, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'New Users',
                    data: [65, 85, 120, 115, 180, 240],
                    borderColor: '#E650C5', // Accent 100
                    backgroundColor: 'rgba(230, 80, 197, 0.1)', // Accent Transparent
                    borderWidth: 2,
                    tension: 0.4, // Curvy line
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#E650C5',
                    pointRadius: 4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: '#E6E6E6' }, // Shaded Gray 10
                        ticks: { color: '#666666', font: { family: 'Aeonik' } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#666666', font: { family: 'Aeonik' } }
                    }
                }
            }
        });

        // --- 2. User Roles Chart (Doughnut) ---
        const ctx2 = document.getElementById('roleChart').getContext('2d');
        new Chart(ctx2, {
            type: 'doughnut',
            data: {
                labels: ['Regular', 'Editors', 'Admins'],
                datasets: [{
                    data: [85, 10, 5],
                    backgroundColor: [
                        '#06134D', // Primary 85
                        '#E650C5', // Accent 100
                        '#FF8E2B', // Happy 100
                    ],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '75%', // Thickness of donut
                plugins: {
                    legend: { display: false }
                }
            }
        });
    </script>
@endsection
