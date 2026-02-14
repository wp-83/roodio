// weekly mood
document.addEventListener('DOMContentLoaded', function() {    
    // get the chart container
    const canvas = document.getElementById('moodChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Get data from window objects
    const moodIcons = window.moodIcons || {};
    const rawData = window.moodWeeklyData || [];
    
    // Sort the data from highest to lowest
    rawData.sort((a, b) => b.total - a.total);
    
    // Save the data into variable
    const moodTypes = rawData.map(item => item.type);
    const moodData = rawData.map(item => item.total);
    
    // Load the images
    const images = {};
    let imagesLoaded = 0;
    const totalImages = moodTypes.length;
    
    if (totalImages === 0) {
        renderChart();
        return;
    }
    
    moodTypes.forEach((mood) => {
        const img = new Image();
        img.src = moodIcons[mood.toLowerCase()];
        images[mood] = img;
        
        img.onload = function() {
            imagesLoaded++;
            if (imagesLoaded === totalImages) {
                renderChart();
            }
        };
        
        img.onerror = function() {
            imagesLoaded++;
            if (imagesLoaded === totalImages) {
                renderChart();
            }
        };
    });
    
    // function for rendering chart
    function renderChart() {
        // remove old chart if exist
        const existingChart = Chart.getChart('moodChart');
        if (existingChart) {
            existingChart.destroy();
        }
        
        // iamge plugin in y-axis
        const imagePlugin = {
            id: 'imagePlugin',
            afterDraw: function(chart) {
                const ctx = chart.ctx;
                const xAxis = chart.scales.x;
                const yAxis = chart.scales.y;
                
                ctx.save();
                
                moodTypes.forEach((mood, index) => {
                    const img = images[mood];
                    
                    if (img && img.complete && img.naturalHeight > 0) {
                        // calculate y position for each image
                        const yPos = yAxis.getPixelForValue(index) - 20;
                        
                        // Background backup for the image
                        ctx.fillStyle = 'transparent';
                        ctx.fillRect(5, yPos - 15, 75, 75);
                        
                        // The image
                        ctx.drawImage(img, 5, yPos - 15, 75, 75);
                    }
                });
                
                ctx.restore();
                
                // grid style
                ctx.save();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 0.5;
                
                // Grid X-axis
                ctx.beginPath();
                const xTicks = xAxis.ticks;
                xTicks.forEach(tick => {
                    const xPos = xAxis.getPixelForValue(tick.value);
                    ctx.moveTo(xPos, yAxis.top);
                    ctx.lineTo(xPos, yAxis.bottom);
                });
                ctx.stroke();
                
                ctx.restore();
            }
        };
        
        // New chart
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: moodTypes.map(() => ''),
                datasets: [{
                    data: moodData,
                    backgroundColor: function(context) {
                        const index = context.dataIndex;
                        const mood = moodTypes[index].toLowerCase();
                        return window.moodColors.default[mood];
                    },
                    hoverBackgroundColor: function(context) {
                        const index = context.dataIndex;
                        const mood = moodTypes[index].toLowerCase();
                        return window.moodColors.hover[mood];
                    },
                    borderRadius: 15,
                    barPercentage: 0.8,
                    categoryPercentage: 0.9
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1500,
                    easing: 'easeInOutQuad'
                },
                layout: {
                    padding: {
                        left: 85
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },

                    tooltip: {
                        backgroundColor: '#ffffff',
                        
                        borderColor: '#A4BEF2',
                        borderWidth: 2.75,
                        displayColors: false,
                        
                        titleColor: '#1F3A98',
                        bodyColor: '#000000',        
                        
                        titleFont: {
                            family: 'Poppins',
                            size: 16,
                            weight: 'bold'
                        },

                        bodyFont: {
                            family: 'Poppins',
                            size: 13,
                            weight: 'normal'
                        },
                        
                        padding: 10,
                        cornerRadius: 10,
                        
                        callbacks: {
                            // tooltip title
                            title: function(context) {
                                const index = context[0].dataIndex;
                                const mood = moodTypes[index];
                                return mood.charAt(0).toUpperCase() + mood.slice(1);
                            },

                            // content
                            label: function(context) {
                                const mood = moodTypes[context.dataIndex];
                                const value = context.raw;
                                
                                return `${value} day(s)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Mood Count',
                            font: { family:'Aeonik', size: 16},
                            color: '#ffffff'
                        },
                        ticks: { 
                            stepSize: 1,
                            color: '#ffffff'
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { display: false }
                    }
                }
            },
            plugins: [imagePlugin]
        });
    }
});