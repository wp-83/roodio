if (!window.HAS_RUN_AUDIO_CONTROL_JS) {
    window.HAS_RUN_AUDIO_CONTROL_JS = true;

    // AUDIO CONTROL JS CONTENT
    const player = document.getElementById('audioPlayer');
    const audio = document.getElementById('audio');
    const prevBtn = document.getElementById('prev');
    const nextBtn = document.getElementById('next');
    const pauseBtn = document.getElementById('pause');
    const playBtn = document.getElementById('play');
    const loopBtn = document.getElementById('loop');
    const shuffleBtn = document.getElementById('shuffle');
    const soundedBtn = document.getElementById('speaker');
    const mutedBtn = document.getElementById('muted');
    const volumeSlider = document.getElementById('volumeSlider');

    // Metadata Elements
    const playerTitle = document.getElementById('playerTitle');
    const playerArtist = document.getElementById('playerArtist');
    const playerImage = document.getElementById('playerImage');

    // Progress bar elements
    const progressBar = document.getElementById("progressBar");
    const currentTimeEl = document.getElementById("currentDuration");
    const durationEl = document.getElementById("duration");
    const progressContainer = document.getElementById('progressContainer');

    const overlayAudioPlay = player ? player.querySelector('#overlayNan') : null;

    if (player && audio) {

        // State
        let playlist = window.currentPlaylist || [];
        let currentIndex = 0;
        let isPlay = false;
        let isMuted = false;
        let isLoop = false;
        let isShuffle = false;
        let hasPlayedOnce = false;

        // ============= HELPER FUNCTIONS =============
        function formatTime(time) {
            if (isNaN(time) || !Number.isFinite(time) || time < 0) return "00:00";
            const minutes = Math.floor(time / 60).toString().padStart(2, '0');
            const seconds = Math.floor(time % 60).toString().padStart(2, "0");
            return `${minutes}:${seconds}`;
        }

        function isSongNan() {
            return (!audio.src || audio.src === '' || audio.src.endsWith('undefined')) ? true : false;
        }

        function isAudioReady() {
            return !isSongNan() && audio.readyState >= 1 && Number.isFinite(audio.duration) && audio.duration > 0;
        }

        function toggleButtonState(btn, isActive) {
            if (!btn) return;
            const activeClass = btn.dataset.activeClass;
            const inactiveClass = btn.dataset.inactiveClass;

            if (activeClass && inactiveClass) {
                if (isActive) {
                    inactiveClass.split(' ').forEach(c => c && btn.classList.remove(c));
                    activeClass.split(' ').forEach(c => c && btn.classList.add(c));
                } else {
                    activeClass.split(' ').forEach(c => c && btn.classList.remove(c));
                    inactiveClass.split(' ').forEach(c => c && btn.classList.add(c));
                }
            }
        }

        function getRandomIndex() {
            if (playlist.length <= 1) return 0;
            // Random index, boleh sama dengan lagu sekarang (true random)
            return Math.floor(Math.random() * playlist.length);
        }

        // ============= MEDIA SESSION API =============
        function setupMediaSession() {
            if ('mediaSession' in navigator) {
                // Set metadata awal
                updateMediaSessionMetadata();
                
                // Set action handlers
                navigator.mediaSession.setActionHandler('play', () => {
                    playAudio();
                });
                
                navigator.mediaSession.setActionHandler('pause', () => {
                    pauseAudio();
                });
                
                navigator.mediaSession.setActionHandler('previoustrack', () => {
                    playPrevious();
                });
                
                navigator.mediaSession.setActionHandler('nexttrack', () => {
                    playNext();
                });
                
                navigator.mediaSession.setActionHandler('seekbackward', (details) => {
                    const skipTime = details.seekOffset || 10;
                    audio.currentTime = Math.max(audio.currentTime - skipTime, 0);
                });
                
                navigator.mediaSession.setActionHandler('seekforward', (details) => {
                    const skipTime = details.seekOffset || 10;
                    audio.currentTime = Math.min(audio.currentTime + skipTime, audio.duration);
                });
                
                navigator.mediaSession.setActionHandler('seekto', (details) => {
                    if (details.fastSeek !== undefined && !isNaN(details.seekTime)) {
                        audio.currentTime = details.seekTime;
                    }
                });
                
                // Update position state
                updatePositionState();
            }
        }

        function updateMediaSessionMetadata() {
            if ('mediaSession' in navigator) {
                const title = playerTitle?.textContent || 'Unknown Title';
                const artist = playerArtist?.textContent || 'Unknown Artist';
                const imageUrl = playerImage?.src || '';
                
                navigator.mediaSession.metadata = new MediaMetadata({
                    title: title,
                    artist: artist,
                    album: 'Roodio',
                    artwork: [
                        {
                            src: imageUrl || '/assets/default-album-art.png',
                            sizes: '96x96',
                            type: 'image/png'
                        },
                        {
                            src: imageUrl || '/assets/default-album-art.png',
                            sizes: '128x128',
                            type: 'image/png'
                        },
                        {
                            src: imageUrl || '/assets/default-album-art.png',
                            sizes: '192x192',
                            type: 'image/png'
                        },
                        {
                            src: imageUrl || '/assets/default-album-art.png',
                            sizes: '256x256',
                            type: 'image/png'
                        },
                        {
                            src: imageUrl || '/assets/default-album-art.png',
                            sizes: '384x384',
                            type: 'image/png'
                        },
                        {
                            src: imageUrl || '/assets/default-album-art.png',
                            sizes: '512x512',
                            type: 'image/png'
                        }
                    ]
                });
            }
        }

        function updatePositionState() {
            if ('mediaSession' in navigator && audio && Number.isFinite(audio.duration) && audio.duration > 0) {
                try {
                    navigator.mediaSession.setPositionState({
                        duration: audio.duration,
                        playbackRate: audio.playbackRate || 1,
                        position: audio.currentTime
                    });
                } catch (e) {
                    // Ignore error if browser doesn't support setPositionState
                }
            }
        }

        // ============= CORE AUDIO FUNCTIONS =============
        function loadSong(index) {
            if (playlist.length == 0) return;

            // Validasi index
            if (index < 0) index = 0;
            if (index >= playlist.length) index = playlist.length - 1;

            currentIndex = index;
            const song = playlist[currentIndex];
            
            // Update audio source
            audio.src = song.path;

            // Sync Metadata
            if (playerTitle) playerTitle.textContent = song.title || 'Unknown Title';
            if (playerArtist) playerArtist.textContent = song.artist || 'Unknown Artist';
            if (playerImage) playerImage.src = song.image || '';

            // Dispatch event for Alpine.js popup and media session
            window.dispatchEvent(new CustomEvent('song-changed', {
                detail: {
                    title: song.title || 'Unknown Title',
                    artist: song.artist || 'Unknown Artist',
                    image: song.image || ''
                }
            }));

            // Update media session metadata
            updateMediaSessionMetadata();

            // Reset progress bar
            if (progressBar) progressBar.style.width = '0%';
            if (currentTimeEl) currentTimeEl.textContent = '00:00';
            if (durationEl) durationEl.textContent = '00:00';

            audio.load();
            
            if (!isSongNan() && overlayAudioPlay) overlayAudioPlay.classList.add('hidden');

            updatePlaylistVisuals(index);
        }

        function updatePlaylistVisuals(index) {
            const allSongs = document.querySelectorAll(`[id^="song-"]`);
            allSongs.forEach(btn => {
                const activeClass = btn.dataset.activeClass;
                const innerDiv = btn.firstElementChild;
                if (activeClass && innerDiv) {
                    activeClass.split(' ').forEach(c => c && innerDiv.classList.remove(c));
                }
            });

            const currentBtn = document.getElementById(`song-${index}`);
            if (currentBtn) {
                const activeClass = currentBtn.dataset.activeClass;
                const innerDiv = currentBtn.firstElementChild;
                if (activeClass && innerDiv) {
                    activeClass.split(' ').forEach(c => c && innerDiv.classList.add(c));
                }
            }
        }

        function playAudio() {
            if (isSongNan()) return;
            
            const playPromise = audio.play();
            if (playPromise !== undefined) {
                playPromise.then(() => {
                    isPlay = true;
                    if (playBtn) playBtn.classList.add('hidden');
                    if (pauseBtn) pauseBtn.classList.remove('hidden');

                    // Update media session state
                    if ('mediaSession' in navigator) {
                        navigator.mediaSession.playbackState = 'playing';
                    }

                    if (!hasPlayedOnce) {
                        hasPlayedOnce = true;
                        window.dispatchEvent(new CustomEvent('open-player-popup'));
                    }
                }).catch(e => console.error("Play error:", e));
            }
        }

        function pauseAudio() {
            audio.pause();
            isPlay = false;
            if (playBtn) playBtn.classList.remove('hidden');
            if (pauseBtn) pauseBtn.classList.add('hidden');

            // Update media session state
            if ('mediaSession' in navigator) {
                navigator.mediaSession.playbackState = 'paused';
            }
        }

        function togglePlay() {
            if (isPlay) {
                pauseAudio();
            } else {
                playAudio();
            }
        }

        function playAfterLoad() {
            const handler = () => {
                playAudio();
                audio.removeEventListener('loadeddata', handler);
            };
            audio.addEventListener('loadeddata', handler, { once: true });
        }

        window.playByIndex = (index) => {
            if (window.currentPlaylist) {
                playlist = window.currentPlaylist;
            }
            loadSong(index);
            audio.addEventListener('loadeddata', function onLoadedData() {
                playAudio();
                audio.removeEventListener('loadeddata', onLoadedData);
            }, { once: true });
        };

        // ============= FIXED: NEXT FUNCTION =============
        function playNext() {
            if (playlist.length === 0) return;
            
            if (isShuffle) {
                // Mode shuffle: pilih random
                if (playlist.length <= 1) {
                    loadSong(0);
                } else {
                    let randomIndex = Math.floor(Math.random() * playlist.length);
                    loadSong(randomIndex);
                }
            } else {
                // Mode normal: next index, loop ke 0 jika di akhir
                let nextIndex = currentIndex + 1;
                if (nextIndex >= playlist.length) {
                    nextIndex = 0; // Loop ke awal playlist
                }
                loadSong(nextIndex);
            }
            playAfterLoad();
        }

        // ============= FIXED: PREV FUNCTION =============
        function playPrevious() {
            if (playlist.length === 0) return;
            
            let prevIndex = currentIndex - 1;
            if (prevIndex < 0) {
                prevIndex = playlist.length - 1; // Loop ke akhir playlist
            }
            loadSong(prevIndex);
            playAfterLoad();
        }

        // ============= EVENT LISTENERS SETUP =============
        
        // Playback controls
        if (nextBtn) {
            nextBtn.addEventListener('click', playNext);
        }

        if (prevBtn) {
            prevBtn.addEventListener('click', playPrevious);
        }

        if (playBtn) playBtn.addEventListener('click', togglePlay);
        if (pauseBtn) pauseBtn.addEventListener('click', togglePlay);

        // LOOP
        if (loopBtn) {
            loopBtn.addEventListener('click', (e) => {
                e.preventDefault();
                isLoop = !isLoop;
                audio.loop = isLoop;
                toggleButtonState(loopBtn, isLoop);
            });
        }

        // SHUFFLE
        if (shuffleBtn) {
            shuffleBtn.addEventListener('click', (e) => {
                e.preventDefault();
                isShuffle = !isShuffle;
                toggleButtonState(shuffleBtn, isShuffle);
            });
        }

        // VOLUME
        if (volumeSlider) {
            let currVol = 1;

            function mutedSound() {
                let tempVol = volumeSlider.value;

                if (!isMuted) {
                    isMuted = true;
                    volumeSlider.value = 0;
                    currVol = tempVol;
                } else {
                    isMuted = false;
                    volumeSlider.value = currVol;
                }

                audio.volume = volumeSlider.value;
                if (soundedBtn) soundedBtn.classList.toggle('hidden');
                if (mutedBtn) mutedBtn.classList.toggle('hidden');
            }

            if (soundedBtn) soundedBtn.addEventListener('click', mutedSound);
            if (mutedBtn) mutedBtn.addEventListener('click', mutedSound);

            volumeSlider.addEventListener('input', () => {
                if (volumeSlider.value == 0) {
                    isMuted = true;
                    if (soundedBtn) soundedBtn.classList.add('hidden');
                    if (mutedBtn) mutedBtn.classList.remove('hidden');
                } else if (soundedBtn && soundedBtn.classList.contains('hidden')) {
                    if (mutedBtn) mutedBtn.classList.add('hidden');
                    if (soundedBtn) soundedBtn.classList.remove('hidden');
                    isMuted = false;
                }

                audio.volume = volumeSlider.value;
            });

            // Set initial volume
            volumeSlider.value = 1;
            audio.volume = 1;
        }

        // PROGRESS BAR EVENT HANDLERS
        if (progressBar && currentTimeEl && durationEl) {
            // Define named handlers
            function handleLoadedMetadata() {
                if (Number.isFinite(audio.duration) && audio.duration > 0) {
                    durationEl.textContent = formatTime(audio.duration);
                    updatePositionState();
                }
            }
            
            function handleTimeUpdate() {
                if (Number.isFinite(audio.duration) && audio.duration > 0) {
                    const percent = (audio.currentTime / audio.duration) * 100;
                    progressBar.style.width = `${percent}%`;
                    currentTimeEl.textContent = formatTime(audio.currentTime);
                    updatePositionState();
                }
            }
            
            // Remove old listeners and add new ones
            audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
            audio.removeEventListener('timeupdate', handleTimeUpdate);
            audio.addEventListener('loadedmetadata', handleLoadedMetadata);
            audio.addEventListener('timeupdate', handleTimeUpdate);
        }

        // ENDED Listener - FIXED
        audio.addEventListener("ended", () => {
            if (!audio.loop) {
                playNext(); // Akan otomatis loop ke 0 jika di akhir
                if (!isPlay) {
                    if (pauseBtn) pauseBtn.classList.add('hidden');
                    if (playBtn) playBtn.classList.remove('hidden');
                }
            }
        });

        // Audio Event Listeners untuk Media Session
        audio.addEventListener('play', () => {
            if ('mediaSession' in navigator) {
                navigator.mediaSession.playbackState = 'playing';
            }
        });

        audio.addEventListener('pause', () => {
            if ('mediaSession' in navigator) {
                navigator.mediaSession.playbackState = 'paused';
            }
        });

        // SEEK CLICK - FIXED
        if (progressContainer) {
            progressContainer.addEventListener("click", (e) => {
                // Validasi lengkap
                if (isSongNan()) {
                    return;
                }
                
                if (audio.readyState < 1) {
                    return;
                }
                
                if (!Number.isFinite(audio.duration) || audio.duration <= 0) {
                    return;
                }

                const rect = progressContainer.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const width = rect.width;

                if (width > 0) {
                    const percentage = clickX / width;
                    const clampedPercentage = Math.max(0, Math.min(1, percentage));
                    const newTime = clampedPercentage * audio.duration;

                    if (Number.isFinite(newTime)) {
                        audio.currentTime = newTime;
                        
                        // Update UI manual untuk respons lebih cepat
                        if (progressBar && currentTimeEl) {
                            const percent = (newTime / audio.duration) * 100;
                            progressBar.style.width = `${percent}%`;
                            currentTimeEl.textContent = formatTime(newTime);
                        }
                    }
                }
            });
        }

        // KEYBOARD SHORTCUTS - FIXED
        document.addEventListener('keydown', (e) => {
            // Skip if no valid audio
            if (isSongNan() || audio.readyState === 0 || playlist.length === 0) return;

            // Space: Play/Pause (jangan mengganggu input field)
            if (e.code === 'Space' && !e.target.matches('input, textarea, [contenteditable]')) {
                e.preventDefault();
                togglePlay();
                return;
            }

            // Arrow keys - dengan penanganan khusus untuk lagu terakhir
            if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && !e.target.matches('input, textarea, [contenteditable]')) {
                // Cek apakah duration valid
                if (!Number.isFinite(audio.duration) || audio.duration <= 0) {
                    return;
                }
                
                e.preventDefault();
                
                if (e.key === 'ArrowRight') {
                    let newTime = audio.currentTime + 10;
                    
                    // Jika melebihi duration, pindah ke next atau wrap
                    if (newTime >= audio.duration) {
                        // Pindah ke lagu berikutnya
                        playNext();
                        return;
                    }
                    
                    if (Number.isFinite(newTime)) {
                        audio.currentTime = newTime;
                        // Update UI
                        if (progressBar && currentTimeEl) {
                            const percent = (newTime / audio.duration) * 100;
                            progressBar.style.width = `${percent}%`;
                            currentTimeEl.textContent = formatTime(newTime);
                        }
                    }
                }

                if (e.key === 'ArrowLeft') {
                    let newTime = audio.currentTime - 10;
                    
                    // Jika kurang dari 0, bisa ke previous atau tetap 0
                    if (newTime < 0) {
                        // Opsi 1: Pindah ke lagu sebelumnya
                        // playPrevious();
                        // return;
                        
                        // Opsi 2: Tetap di 0 (default)
                        newTime = 0;
                    }
                    
                    if (Number.isFinite(newTime)) {
                        audio.currentTime = newTime;
                        // Update UI
                        if (progressBar && currentTimeEl) {
                            const percent = (newTime / audio.duration) * 100;
                            progressBar.style.width = `${percent}%`;
                            currentTimeEl.textContent = formatTime(newTime);
                        }
                    }
                }
            }
        });

        // POPUP BEHAVIOR
        function popupBehaviour(element) {
            if (element.classList.contains('invisible')) {
                element.classList.remove('opacity-0', 'invisible');
            } else {
                element.classList.add('opacity-0', 'invisible');
            }
        }

        const audioCtrlArea = document.getElementById('audioControlResponsive');
        const audioCtrlPopup = document.getElementById('audioControlPopup');

        if (audioCtrlArea && audioCtrlPopup) {
            const audioCtrlContent = audioCtrlPopup.querySelector('.popupContent');

            audioCtrlArea.addEventListener('click', () => {
                popupBehaviour(audioCtrlPopup);
            });

            document.addEventListener('mousedown', (e) => {
                if (audioCtrlContent && !audioCtrlContent.contains(e.target) && !audioCtrlArea.contains(e.target)) {
                    audioCtrlContent.classList.add('opacity-0', 'invisible');
                    audioCtrlPopup.classList.add('opacity-0', 'invisible');
                }
            });
        }

        // ============= INITIALIZATION =============
        document.addEventListener('DOMContentLoaded', () => {
            if (playlist && playlist.length > 0) {
                if (isSongNan()) {
                    loadSong(0);
                }
            }

            if (overlayAudioPlay) {
                if (isSongNan()) {
                    overlayAudioPlay.classList.remove('hidden');
                } else {
                    overlayAudioPlay.classList.add('hidden');
                }
            }

            // Setup Media Session
            setupMediaSession();
        });

        // Listen for song changes to update media session
        window.addEventListener('song-changed', () => {
            updateMediaSessionMetadata();
        });

        // Debugging events (optional)
        audio.addEventListener('error', (e) => {
            console.error('Audio error:', e);
        });
    }
}