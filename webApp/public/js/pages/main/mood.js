// weekly mood
document.addEventListener('DOMContentLoaded', function() {    
    // get the chart container
    const canvas = document.getElementById('moodChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Get data from window objects
    const moodIcons = window.moodIcons || {};
    const rawData = window.moodWeeklyData || [];
    
    if (rawData.length === 0) return;
    
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
        
        // image plugin in y-axis
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
                        const yPos = yAxis.getPixelForValue(index) - 20;
                        ctx.drawImage(img, 5, yPos - 15, 75, 75);
                    }
                });
                
                ctx.restore();
                
                // grid style
                ctx.save();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 0.5;
                
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
                    categoryPercentage: 0.9,
                    
                    animation: {
                        duration: 1500,
                        easing: 'easeInOutQuad'
                    }
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuad',
                    
                    mode: 'all',
                    
                    onProgress: function(context) {
                        if (context.chart) {
                            context.chart.draw();
                        }
                    }
                },
                
                transitions: {
                    show: {
                        animation: {
                            duration: 1750,
                            easing: 'easeInOutQuad'
                        }
                    },
                    active: {
                        animation: {
                            duration: 300
                        }
                    },
                    resize: {
                        animation: {
                            duration: 0
                        }
                    }
                },
                
                layout: {
                    padding: {
                        left: 85
                    }
                },
                
                elements: {
                    bar: {
                        backgroundColor: 'transparent'
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
                            title: function(context) {
                                const index = context[0].dataIndex;
                                const mood = moodTypes[index];
                                return mood.charAt(0).toUpperCase() + mood.slice(1);
                            },

                            label: function(context) {
                                return `${context.raw} day(s)`;
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
                        },

                        grid: {
                            color: 'rgba(255,255,255,0.2)'
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

// monthly mood
document.addEventListener('DOMContentLoaded', function () {
    // Get today's date for current month
    var today = new Date();
    var currentYear = today.getFullYear();
    var currentMonth = today.getMonth();
    
    // Month names in English
    var monthNames = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ];
    var currentMonthName = monthNames[currentMonth];

    // ===== ISI ELEMENT MONTH DAN YEAR =====
    var monthEl = document.getElementById('month');
    var yearEl = document.getElementById('year');
    
    monthEl.innerHTML = currentMonthName;
    yearEl.innerHTML = currentYear;

    // Make sure calendar element exists
    var calendarEl = document.getElementById('calendar');
    if (!calendarEl) return;

    // Get calendar data from window object
    var calendarData = window.calendarData || [];

    // Calculate dominant mood for the month
    function calculateDominantMood(data) {
        if (!data || data.length === 0) {
            return {
                type: 'No Data',
                total: 0,
                percentage: "0.00",
                icon: null
            };
        }

        var moodCounts = {};
        data.forEach(function(item) {
            var mood = item.type;
            if (moodCounts[mood]) {
                moodCounts[mood] += item.total || 1;
            } else {
                moodCounts[mood] = item.total || 1;
            }
        });

        var dominantMood = null;
        var maxCount = 0;
        
        for (var mood in moodCounts) {
            if (moodCounts[mood] > maxCount) {
                maxCount = moodCounts[mood];
                dominantMood = mood;
            }
        }

        var totalAllMoods = Object.values(moodCounts).reduce((a, b) => a + b, 0);
        var percentage = totalAllMoods > 0 ? (maxCount / totalAllMoods * 100).toFixed(2) : "0.00";

        return {
            type: dominantMood,
            total: maxCount,
            percentage: percentage,
            icon: window.moodIcons ? window.moodIcons[dominantMood] : null
        };
    }

    var dominantMood = calculateDominantMood(calendarData);
    var summaryEl = document.getElementById('calendarSummary');
    
    if (summaryEl) {
        const moodEl = summaryEl.querySelector("#moodDominant");
        const totalEl = summaryEl.querySelector("#totalMoodDominant");
        const percentageEl = summaryEl.querySelector("#percentageMoodDominant");
        const imgEl = summaryEl.querySelector("#dominantMoodImage");
        
        if (moodEl) moodEl.innerHTML = dominantMood.type.toUpperCase();
        if (totalEl) totalEl.innerHTML = dominantMood.total;
        if (percentageEl) percentageEl.innerHTML = dominantMood.percentage + '%';
        if (imgEl) imgEl.src = window.baseUrl + 'assets/moods/icons/' + dominantMood.type.toLowerCase() + '.png';
    }
    
    var style = document.createElement('style');
    style.textContent = `
        /* ===== RESET ALL BACKGROUNDS ===== */
        .fc,
        .fc *,
        .fc-scrollgrid,
        .fc-scrollgrid *,
        .fc-daygrid-day,
        .fc-daygrid-day *,
        .fc-col-header-cell,
        .fc-col-header-cell *,
        .fc-daygrid-body,
        .fc-daygrid-body *,
        .fc-daygrid-day-frame,
        .fc-daygrid-day-events,
        .fc-daygrid-day-top,
        .fc-daygrid-day-bottom,
        .fc-daygrid-event,
        .fc-event,
        .fc-event *,
        table, tr, td, th, tbody, thead {
            background: transparent !important;
            background-color: transparent !important;
            outline: none !important;
        }
        
        /* ===== BACKGROUND KOTAK UNTUK SEMUA CELL ===== */
        .fc-daygrid-day {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
            margin: 2px !important;
            transition: background-color 0.2s ease !important;
        }
        
        /* ===== HIGHLIGHT UNTUK TANGGAL AKTIF (HARI INI) ===== */
        .fc-day-today {
            background-color: rgba(164, 190, 242, 0.25) !important;
            border: 2px solid #A4BEF2 !important;
            box-shadow: 0 0 10px rgba(164, 190, 242, 0.3) !important;
        }
        
        /* ===== RESPONSIVE CONTAINER ===== */
        #calendar {
            width: 100% !important;
            height: auto !important;
        }
        
        .fc {
            width: 100% !important;
            height: auto !important;
            max-width: 100% !important;
        }
        
        .fc-view-harness {
            height: auto !important;
        }
        
        .fc-scrollgrid {
            width: 100% !important;
            table-layout: fixed !important;
            border-collapse: separate !important;
            border-spacing: 2px !important;
        }
        
        /* ===== HEADER STATIC - TAPI TETAP JADI TABLE ===== */
        .fc-col-header {
            width: 100% !important;
            background: transparent !important;
        }
        
        .fc-col-header-cell {
            border-bottom: 2px solid rgba(255, 255, 255, 0.3) !important;
            background: transparent !important;
            height: 45px !important;
            vertical-align: middle !important;
            text-align: center !important;
            padding: 0 !important;
        }
        
        .fc-col-header-cell-cushion {
            color: white !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            font-family: 'Poppins'!important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            padding: 12px 0 !important;
            display: inline-block !important;
            text-decoration: none !important;
        }
        
        /* ===== SEMBUNYIKAN BARIS KOSONG ===== */
        .fc-daygrid-body > table > tbody > tr:has(> td > .fc-day-other:only-child) {
            display: none !important;
        }
        
        .fc-daygrid-body > table > tbody > tr:last-child:has(.fc-day-other) {
            display: none !important;
        }
        
        .fc-daygrid-body > table > tbody > tr:first-child:has(.fc-day-other) {
            display: none !important;
        }
        
        .fc-daygrid-body > table > tbody > tr:has(.fc-daygrid-day:not(.fc-day-other)) {
            display: table-row !important;
        }
        
        /* ===== BORDER MENGGUNAKAN PSEUDO-ELEMENTS ===== */
        .fc-daygrid-day {
            position: relative !important;
        }

        .fc-daygrid-day-bg {
            height: 0px !important;
        }
        
        .fc-daygrid-day::after {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            right: 0 !important;
            width: 1px !important;
            height: 100% !important;
            background-color: rgba(255, 255, 255, 0.2) !important;
            z-index: 1000 !important;
            pointer-events: none !important;
        }
        
        .fc-daygrid-day::before {
            content: '' !important;
            position: absolute !important;
            bottom: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 1px !important;
            background-color: rgba(255, 255, 255, 0.2) !important;
            z-index: 1000 !important;
            pointer-events: none !important;
        }
        
        .fc-daygrid-day:last-child::after {
            display: none !important;
        }
        
        .fc-daygrid-body tr:last-child .fc-daygrid-day::before {
            display: none !important;
        }

        /* ===== DAY NUMBERS - TANPA HIGHLIGHT ===== */
        .fc-daygrid-day:not(.fc-day-other) .fc-daygrid-day-number {
            display: block !important;
            color: white !important;
            font-size: 1rem !important;
            font-family: 'Poppins', sans-serif !important;
            padding: 4px !important;
            text-decoration: none !important;
            text-align: center !important;
            background: transparent !important;
            border-radius: 0 !important;
        }
        
        .fc-day-other .fc-daygrid-day-number,
        .fc-day-other .fc-daygrid-day-events {
            display: none !important;
        }
        
        .fc-day-other {
            visibility: visible !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        
        /* ===== EVENT STYLING - RESPONSIF ===== */
        .fc-event {
            background: transparent !important;
            border: none !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            height: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .fc-event img {
            width: 90px !important;
            height: 90px !important;
            object-fit: contain !important;
            margin: 0 auto !important;
            display: block !important;
        }
        
        .fc-daygrid-day-events .fc-event:only-child img {
            width: 110px !important;
            height: 110px !important;
        }

        /* ===== HAPUS SEMUA BORDER LUAR ===== */
        .fc-scrollgrid, 
        .fc-scrollgrid *,
        .fc-daygrid-body table,
        .fc-daygrid-body table * {
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: none !important;
            border-width: 0 !important;
        }

        /* Pastikan border dalam (pseudo-elements) tetap jalan */
        .fc-daygrid-day::after,
        .fc-daygrid-day::before {
            display: block !important;
        }
        
        /* ===== CELL SIZING ===== */
        .fc-daygrid-day-frame {
            min-height: 160px !important;
            height: auto !important;
            display: flex !important;
            flex-direction: column !important;
            padding: 4px !important;
        }
        
        .fc-daygrid-day-events {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            gap: 4px !important;
            min-height: 100px !important;
        }

        /* ===== TOOLTIP CUSTOM-WHITE ===== */
        .tippy-box[data-theme~='custom-white'] {
            background-color: #ffffff !important;
            color: #333333 !important;
            border: 1px solid #dddddd !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        }

        .tippy-box[data-theme~='custom-white'] .tippy-arrow {
            color: #ffffff !important;
        }

        .tippy-box[data-theme~='custom-white'] .tippy-content {
            padding: 8px 12px !important;
            font-family: 'Poppins', sans-serif !important;
        }

        .tippy-box[data-theme~='custom-white'] strong {
            color: #1F3A98 !important;
        }
        
        /* ===== RESPONSIVE BREAKPOINTS ===== */
        @media (min-width: 1200px) {
            .fc-event img {
                width: 90px !important;
                height: 90px !important;
            }
            .fc-daygrid-day-events .fc-event:only-child img {
                width: 110px !important;
                height: 110px !important;
            }
            .fc-daygrid-day-frame {
                min-height: 160px !important;
            }
        }
        
        @media (min-width: 992px) and (max-width: 1199px) {
            .fc-event img {
                width: 75px !important;
                height: 75px !important;
            }
            .fc-daygrid-day-events .fc-event:only-child img {
                width: 90px !important;
                height: 90px !important;
            }
            .fc-daygrid-day-frame {
                min-height: 140px !important;
            }
        }
        
        @media (min-width: 768px) and (max-width: 991px) {
            .fc-event img {
                width: 60px !important;
                height: 60px !important;
            }
            .fc-daygrid-day-events .fc-event:only-child img {
                width: 75px !important;
                height: 75px !important;
            }
            .fc-daygrid-day-frame {
                min-height: 130px !important;
            }
        }
        
        @media (max-width: 767px) {
            .fc-event img {
                width: 45px !important;
                height: 45px !important;
            }
            .fc-daygrid-day-events .fc-event:only-child img {
                width: 55px !important;
                height: 55px !important;
            }
            .fc-daygrid-day-frame {
                min-height: 110px !important;
            }
        }
        
        @media (max-width: 480px) {
            .fc-event img {
                width: 35px !important;
                height: 35px !important;
            }
            .fc-daygrid-day-events .fc-event:only-child img {
                width: 45px !important;
                height: 45px !important;
            }
            .fc-daygrid-day-frame {
                min-height: 90px !important;
            }
        }
    `;
    document.head.appendChild(style);

    // Initialize calendar
    var calendar = new FullCalendar.Calendar(
        calendarEl,
        {
            initialView: 'dayGridMonth',
            height: "auto",
            contentHeight: "auto",
            aspectRatio: 1.2,
            initialDate: today,
            
            // MATIKAN STICKY HEADER
            headerToolbar: false,
            stickyHeaderDates: false,
            
            datesSet: function(info) {
                if (info.view.currentStart.getMonth() !== currentMonth || 
                    info.view.currentStart.getFullYear() !== currentYear) {
                    
                    setTimeout(function() {
                        calendar.gotoDate(new Date(currentYear, currentMonth, 1));
                    }, 10);
                }
                
                setTimeout(function() {
                    document.querySelectorAll('.fc-daygrid-body table tbody tr').forEach(function(row) {
                        var hasCurrentMonthCell = false;
                        row.querySelectorAll('.fc-daygrid-day').forEach(function(cell) {
                            if (!cell.classList.contains('fc-day-other')) {
                                hasCurrentMonthCell = true;
                            }
                        });
                        
                        if (!hasCurrentMonthCell) {
                            row.style.display = 'none';
                        }
                    });
                }, 100);
            },
            
            events: calendarData,

            eventDidMount: function(info) {
                var mood = info.event.extendedProps.type;
                var total = info.event.extendedProps.total;
                
                info.el.innerHTML = '';
                
                if (mood && window.moodIcons && window.moodIcons[mood]) {
                    var img = document.createElement('img');
                    img.src = window.moodIcons[mood];
                    img.alt = mood;
                    
                    img.style.display = 'block';
                    img.style.margin = '0 auto';
                    img.style.objectFit = 'contain';
                    
                    info.el.appendChild(img);
                    
                    img.onerror = function() {
                        info.el.innerHTML = mood;
                    };
                } else {
                    info.el.innerHTML = mood || 'event';
                }
                
                if (typeof tippy !== 'undefined') {
                    tippy(info.el, {
                        content: `
                            <div style="text-align:center; font-size:1rem;">
                                <strong>${info.event.title || mood}</strong><br>
                            </div>
                        `,
                        theme: 'custom-white',
                        allowHTML: true,
                        animation: 'scale',
                        placement: 'top'
                    });
                }
            },
            
            dayCellDidMount: function(info) {
                info.el.style.cursor = 'default';
            }
        }
    );

    calendar.render();
    
    setTimeout(function() {
        document.querySelectorAll('.fc-daygrid-body table tbody tr').forEach(function(row) {
            var hasCurrentMonthCell = false;
            row.querySelectorAll('.fc-daygrid-day').forEach(function(cell) {
                if (!cell.classList.contains('fc-day-other')) {
                    hasCurrentMonthCell = true;
                }
            });
            
            if (!hasCurrentMonthCell) {
                row.style.display = 'none';
            }
        });
    }, 200);
});