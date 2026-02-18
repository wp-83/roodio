@extends('layouts.admin.master')

@section('content')
<div class='flex flex-col w-full h-full p-6'>
    <div class='flex flex-col w-full h-fit mb-5'>
        <h1 class='text-4xl font-primary font-bold text-white mb-3'>Model Monitoring</h1>
        <p class='text-secondary text-sm md:text-base'>Real-time mood prediction performance metrics.</p>
    </div>

    <!-- Alert for Data Drift -->
    @if($avgConfidence < 60)
    <div class="w-full bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-5" role="alert">
        <strong class="font-bold">Warning: Data Drift Detected!</strong>
        <span class="block sm:inline">The average confidence score has dropped below 60%. Retraining may be required.</span>
    </div>
    @endif

    <div class='grid grid-cols-1 md:grid-cols-3 gap-6 mb-8'>
        <!-- Overall Accuracy Card -->
        <div class='bg-primary-90 rounded-2xl p-6'>
            <h3 class='text-secondary text-sm font-bold uppercase tracking-wider mb-2'>Accuracy</h3>
            <p class='text-4xl font-bold text-white'>{{ number_format($overallAccuracy, 1) }}%</p>
            <p class='text-xs text-secondary mt-1'>Based on {{ $totalFeedbacks }} user feedbacks</p>
        </div>
        
        <!-- Confidence Score Card -->
        <div class='bg-primary-90 rounded-2xl p-6'>
            <h3 class='text-secondary text-sm font-bold uppercase tracking-wider mb-2'>Avg Confidence</h3>
            <p class='text-4xl font-bold {{ $avgConfidence < 60 ? "text-red-500" : "text-white" }}'>
                {{ number_format($avgConfidence, 1) }}%
            </p>
            <p class='text-xs text-secondary mt-1'>Threshold: 60%</p>
        </div>

        <!-- Mispredicted Count Card -->
        <div class='bg-primary-90 rounded-2xl p-6'>
            <h3 class='text-secondary text-sm font-bold uppercase tracking-wider mb-2'>Mispredictions</h3>
            <p class='text-4xl font-bold text-red-400'>{{ count($mispredictedSongs) }}</p>
            <p class='text-xs text-secondary mt-1'>Candidates for retraining</p>
        </div>
    </div>

    <div class='grid grid-cols-1 lg:grid-cols-2 gap-8'>
        <!-- Accuracy Trend Chart -->
        <div class='bg-primary-90 rounded-2xl p-6'>
            <h3 class='text-secondary text-sm font-bold uppercase tracking-wider mb-4'>Accuracy Trend (Last 30 Days)</h3>
            <div class='w-full h-64 relative'>
                 <canvas id="accuracyChart"></canvas>
            </div>
        </div>

        <!-- Retraining Candidates List -->
        <div class='bg-primary-90 rounded-2xl p-6 overflow-hidden'>
            <h3 class='text-secondary text-sm font-bold uppercase tracking-wider mb-4'>Retraining Candidates</h3>
            <div class='overflow-y-auto h-64 pr-2 custom-scrollbar'>
                @forelse($mispredictedSongs as $item)
                <div class='flex items-center gap-3 mb-4 last:mb-0 p-3 bg-primary-80 rounded-xl'>
                    <img src="{{ $item->song->photoPath ? config('filesystems.disks.azure.url') . '/' . $item->song->photoPath : asset('assets/images/placeholder.jpg') }}" 
                         class='w-10 h-10 rounded object-cover'>
                    <div class='flex-1 min-w-0'>
                        <p class='text-white font-bold truncate'>{{ $item->song->title ?? 'Unknown Song' }}</p>
                        <p class='text-xs text-secondary truncate'>
                            Predicted: <span class='text-warning'>{{ $item->song->modelLog->predicted_mood ?? 'N/A' }}</span> 
                            ({{ number_format($item->song->modelLog->confidence_score ?? 0, 2) }})
                        </p>
                    </div>
                    <div class='shrink-0 bg-red-500/20 px-2 py-1 rounded text-[10px] text-red-400 font-bold'>
                        {{ $item->negative_count }} reports
                    </div>
                </div>
                @empty
                <p class='text-secondary text-sm'>No mispredictions recorded yet.</p>
                @endforelse
            </div>
        </div>
    </div>
</div>
@endsection

@section('scripts')
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        const accuracyData = @json($accuracyTrend);
        
        if(accuracyData.length > 0) {
            const ctx = document.getElementById('accuracyChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: accuracyData.map(d => d.date),
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: accuracyData.map(d => d.accuracy),
                        borderColor: '#22c55e', // Green-500
                        backgroundColor: 'rgba(34, 197, 94, 0.2)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#9ca3af' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#9ca3af' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
    });
</script>
@endsection
