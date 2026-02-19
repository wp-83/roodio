document.addEventListener('DOMContentLoaded', function () {
    // 1. Script untuk Audio Upload Preview Filename
    const songInput = document.getElementById('song-input');
    if (songInput) {
        songInput.addEventListener('change', function (e) {
            var fileName = e.target.files[0] ? e.target.files[0].name : "Browse Audio File";
            const fileNameText = document.getElementById('audio-filename');
            const placeholder = document.getElementById('audio-placeholder');

            fileNameText.textContent = fileName;

            if (e.target.files[0]) {
                fileNameText.classList.remove('text-shadedOfGray-30');
                fileNameText.classList.add('text-secondary-happy-100', 'font-bold');
                // Ganti icon jadi check file
                placeholder.querySelector('i').className = "fa-solid fa-file-audio text-2xl text-secondary-happy-100 mb-2";
            } else {
                fileNameText.classList.add('text-shadedOfGray-30');
                fileNameText.classList.remove('text-secondary-happy-100', 'font-bold');
                placeholder.querySelector('i').className = "fa-solid fa-cloud-arrow-up text-2xl text-primary-20 mb-2";
            }
        });
    }

    // 2. Script untuk Photo Upload Preview
    const photoInput = document.getElementById('photo_input');
    if (photoInput) {
        photoInput.addEventListener('change', function (e) {
            const file = e.target.files[0];
            const previewElement = document.getElementById('photo-preview');
            const placeholderElement = document.getElementById('photo-placeholder');
            const overlayElement = document.getElementById('preview-overlay');

            if (file) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    previewElement.src = e.target.result;
                    previewElement.classList.remove('hidden');
                    placeholderElement.classList.add('opacity-0', 'absolute');
                    overlayElement.classList.remove('hidden');
                    overlayElement.classList.add('flex');
                }
                reader.readAsDataURL(file);
            } else {
                previewElement.src = '#';
                previewElement.classList.add('hidden');
                placeholderElement.classList.remove('opacity-0', 'absolute');
                overlayElement.classList.add('hidden');
                overlayElement.classList.remove('flex');
            }
        });
    }

    // 3. Script untuk Async Upload Form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function (e) {
            e.preventDefault();

            // Ambil elemen overlay
            const overlay = document.getElementById('upload-overlay');
            const progressBar = document.getElementById('upload-progress-bar');
            const percentageText = document.getElementById('upload-percentage');
            const statusTitle = document.getElementById('upload-status-title');
            const statusText = document.getElementById('upload-status-text');

            // Reset state
            progressBar.style.width = '0%';
            percentageText.textContent = '0%';
            statusTitle.textContent = 'Uploading...';
            statusText.textContent = 'Please wait while we upload your track.';

            // Tampilkan Overlay
            // Ubah dari fixed ke absolute agar berada di dalam container, bukan full screen
            overlay.classList.remove('hidden');
            overlay.classList.add('flex');

            // Pastikan parent (main wrapper) memiliki position: relative jika belum
            // Namun karena layout master sudah punya relative pada content wrapper, ini harusnya aman
            // Kita juga sesuaikan styling overlay agar pas di tengah konten
            overlay.style.position = 'absolute';
            overlay.style.zIndex = '100';

            const formData = new FormData(this);
            let progressInterval;

            // Function to update UI
            const updateProgress = (percent) => {
                progressBar.style.width = percent + '%';
                percentageText.textContent = Math.round(percent) + '%';
            };

            axios.post(this.action, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
                },
                onUploadProgress: function (progressEvent) {
                    // Scale upload progress to max 80%
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    const scaledProgress = Math.min(80, percentCompleted * 0.8);

                    updateProgress(scaledProgress);

                    if (percentCompleted >= 100) {
                        statusTitle.textContent = 'Analyzing Mood...';
                        statusText.textContent = 'Our AI is listening to your track to determine its mood...';

                        // Start simulated progress from 80% to 99%
                        let currentProgress = 80;
                        progressInterval = setInterval(() => {
                            if (currentProgress < 99) {
                                currentProgress += 0.5; // Increment slowly
                                updateProgress(currentProgress);
                            }
                        }, 200); // Update every 200ms
                    }
                }
            })
                .then(response => {
                    clearInterval(progressInterval);
                    updateProgress(100);

                    if (response.data.success) {
                        statusTitle.textContent = 'Success!';
                        statusText.textContent = 'Redirecting...';

                        setTimeout(() => {
                            // Gunakan redirect dari response, tapi jika session flash hilang karena XHR,
                            // kita bisa memaksa reload atau membiarkan Laravel handle session. 
                            // Biasanya window.location.href cukup.
                            // Jika flash message tidak muncul, kemungkinan karena request sebelumnya adalah AJAX.
                            // Kita bisa passing parameter success di URL sebagai fallback.
                            let redirectUrl = response.data.redirect;
                            // Cek apakah url sudah punya query param
                            redirectUrl += (redirectUrl.includes('?') ? '&' : '?') + 'success_upload=true';

                            window.location.href = redirectUrl;
                        }, 500);
                    }
                })
                .catch(error => {
                    clearInterval(progressInterval);

                    // Sembunyikan Overlay
                    overlay.classList.add('hidden');
                    overlay.classList.remove('flex');

                    // Remove pulse animation
                    progressBar.closest('.relative').classList.remove('animate-pulse');

                    if (error.response && error.response.status === 422) {
                        let message = error.response.data.message || 'Validation Error. Please check your inputs.';
                        showErrorModal(message);
                    } else {
                        showErrorModal('An error occurred. Please try again or check your connection.');
                    }
                });
        });
    }
});

// Helper function to show error modal
function showErrorModal(message) {
    const errorModal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message-text');

    if (errorModal && errorMessage) {
        errorMessage.textContent = message;
        errorModal.classList.remove('hidden');
        errorModal.classList.add('flex');

        // Ensure error modal is positioned correctly (absolute like overlay)
        errorModal.style.position = 'absolute';
        errorModal.style.zIndex = '110'; // Higher than overlay
    } else {
        alert(message); // Fallback
    }
}
