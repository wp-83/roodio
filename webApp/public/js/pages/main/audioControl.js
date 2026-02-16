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
    const loopMobileBtn = document.getElementById('loop-mobile');
    const shuffleMobileBtn = document.getElementById('shuffle-mobile');
    const speakerMobileBtn = document.getElementById('speaker-mobile');

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
        let playOrder = []; // track indices in play order

        // Build play order array (normal or shuffled)
        function buildPlayOrder() {
            playOrder = Array.from({ length: playlist.length }, (_, i) => i);
            if (isShuffle) {
                // Fisher-Yates shuffle, but keep currentIndex at top
                for (let i = playOrder.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [playOrder[i], playOrder[j]] = [playOrder[j], playOrder[i]];
                }
                // Move current song to top of the list
                const curPos = playOrder.indexOf(currentIndex);
                if (curPos > 0) {
                    playOrder.splice(curPos, 1);
                    playOrder.unshift(currentIndex);
                }
            }
        }

        // Render tracks in the popup panel
        function renderPopupTracks() {
            const container = document.getElementById('popupTracksList');
            if (!container) return;

            container.innerHTML = '';
            if (playlist.length === 0) {
                container.innerHTML = '<p class="text-white font-secondaryAndButton font-bold text-small lg:text-body-size text-center py-4">No tracks in queue</p>';
                return;
            }

            playOrder.forEach((songIdx, orderIdx) => {
                const song = playlist[songIdx];
                if (!song) return;
                const isActive = songIdx === currentIndex;

                const item = document.createElement('div');
                item.className = `flex flex-row items-center gap-3 px-2 py-1.5 rounded-lg cursor-pointer transition-colors duration-150 ${isActive ? 'bg-primary-70/70' : 'hover:bg-primary-85'
                    }`;
                item.dataset.songIndex = songIdx;

                item.innerHTML = `
                    <span class="text-small font-secondaryAndButton ${isActive ? 'text-white' : 'text-shadedOfGray-30'} w-6 text-center shrink-0">${orderIdx + 1}</span>
                    <div class="w-10 h-10 rounded-md overflow-hidden bg-primary-70 shrink-0">
                        <img src="${song.cover || song.image || ''}" alt="song" class="w-full h-full object-cover">
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="text-small lg:text-body-size font-secondaryAndButton truncate ${isActive ? 'text-white font-bold' : 'text-white'}">${song.title || 'Unknown'}</p>
                        <p class="text-micro lg:text-small truncate font-secondaryAndButton text-shadedOfGray-30">${song.artist || 'Unknown'}</p>
                    </div>
                `;

                item.addEventListener('click', () => {
                    window.playByIndex(songIdx);
                });

                container.appendChild(item);
            });
        }

        // Render lyrics for the current song in the popup panel
        function renderPopupLyrics() {
            const container = document.getElementById('popupLyricsContent');
            if (!container) return;

            if (playlist.length === 0 || !playlist[currentIndex]) {
                container.textContent = 'No lyrics available';
                return;
            }

            const song = playlist[currentIndex];
            const lyrics = song.lyrics || '';

            if (!lyrics || lyrics.trim() === '') {
                container.innerHTML = '<p class="text-primary-40 text-center py-4">No lyrics available for this song</p>';
            } else {
                // Escape HTML and preserve line breaks
                const escaped = lyrics.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                container.textContent = '';
                container.innerText = lyrics;
            }
        }

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
            if (playerImage) playerImage.src = song.cover || song.image || '';

            // Dispatch event for Alpine.js popup and media session
            window.dispatchEvent(new CustomEvent('song-changed', {
                detail: {
                    title: song.title || 'Unknown Title',
                    artist: song.artist || 'Unknown Artist',
                    image: song.cover || song.image || ''
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
            renderPopupTracks();
            renderPopupLyrics();
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

        // ============= NEXT FUNCTION =============
        function playNext() {
            if (playlist.length === 0) return;

            // Find current position in playOrder
            const currentPos = playOrder.indexOf(currentIndex);
            let nextPos = currentPos + 1;
            if (nextPos >= playOrder.length) {
                nextPos = 0; // Loop ke awal
            }
            loadSong(playOrder[nextPos]);
            playAfterLoad();
        }

        // ============= PREV FUNCTION =============
        function playPrevious() {
            if (playlist.length === 0) return;

            // Find current position in playOrder
            const currentPos = playOrder.indexOf(currentIndex);
            let prevPos = currentPos - 1;
            if (prevPos < 0) {
                prevPos = playOrder.length - 1; // Loop ke akhir
            }
            loadSong(playOrder[prevPos]);
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
        function toggleLoop() {
            isLoop = !isLoop;
            audio.loop = isLoop;
            toggleButtonState(loopBtn, isLoop);
            toggleButtonState(loopMobileBtn, isLoop);
        }

        if (loopBtn) {
            loopBtn.addEventListener('click', (e) => {
                e.preventDefault();
                toggleLoop();
            });
        }
        if (loopMobileBtn) {
            loopMobileBtn.addEventListener('click', (e) => {
                e.preventDefault();
                toggleLoop();
            });
        }

        // SHUFFLE
        function toggleShuffle() {
            isShuffle = !isShuffle;
            toggleButtonState(shuffleBtn, isShuffle);
            toggleButtonState(shuffleMobileBtn, isShuffle);
            buildPlayOrder();
            renderPopupTracks();
        }

        if (shuffleBtn) {
            shuffleBtn.addEventListener('click', (e) => {
                e.preventDefault();
                toggleShuffle();
            });
        }
        if (shuffleMobileBtn) {
            shuffleMobileBtn.addEventListener('click', (e) => {
                e.preventDefault();
                toggleShuffle();
            });
        }

        // ============= SLIDER & VOLUME =============

        // SLEEP TIMER LOGIC
        let sleepTimerId = null;
        let sleepMode = null; // 'time', 'end', or null
        let currentSleepMinutes = null; // Store the active duration

        const sleepTimerBtn = document.getElementById('sleep-timer'); // Desktop trigger
        const sleepTimerMobileBtn = document.getElementById('sleep-timer-mobile'); // Mobile trigger (Accordion Toggle)
        const sleepTimerStatusMobile = document.getElementById('sleep-timer-status-mobile'); // Mobile Status Text
        const sleepTimerOptions = document.querySelectorAll('.sleep-timer-opt'); // Desktop dropdown options
        const sleepTimerOptionsMobile = document.querySelectorAll('.sleep-timer-opt-mobile'); // Mobile dropdown options (New)
        const audioSleepTimerPopup = document.getElementById('audioSleepTimerPopup');

        function updateSleepTimerUI() {
            // Update Active State on Desktop Button
            if (sleepTimerBtn) {
                toggleButtonState(sleepTimerBtn, sleepMode !== null);
            }

            // Update Mobile Button Active State
            if (sleepTimerMobileBtn) {
                toggleButtonState(sleepTimerMobileBtn, sleepMode !== null);
            }

            // Update Status Text on Mobile
            if (sleepTimerStatusMobile) {
                if (sleepMode === 'end') {
                    sleepTimerStatusMobile.textContent = 'End of Track';
                    sleepTimerStatusMobile.classList.remove('hidden');
                } else if (sleepMode === 'time') {
                    sleepTimerStatusMobile.textContent = `${currentSleepMinutes} Minutes`;
                    sleepTimerStatusMobile.classList.remove('hidden');
                } else {
                    sleepTimerStatusMobile.classList.add('hidden');
                }
            }

            // Helper to update indicators and styles
            const updateIndicators = (options, isMobile) => {
                options.forEach(opt => {
                    const val = opt.dataset.value;
                    const indicator = opt.querySelector(isMobile ? '.active-indicator-mobile' : '.active-indicator');

                    let isActive = false;

                    if (sleepMode === 'end' && val === 'end') isActive = true;
                    if (sleepMode === null && val === 'off') isActive = true; // Optional: highlight "Turn Off" when off
                    if (sleepMode === 'time' && parseInt(val) === currentSleepMinutes) isActive = true;

                    // Apply/Remove Active Stylings matching Hover
                    if (isActive) {
                        if (indicator) indicator.classList.remove('hidden');

                        // Active Styling
                        if (isMobile) {
                            opt.classList.add('text-blue-500', 'font-bold');
                            opt.classList.remove('text-primary-40');
                        } else {
                            opt.classList.add('bg-primary-70', 'text-white');
                            opt.classList.remove('text-gray-300');
                        }
                    } else {
                        if (indicator) indicator.classList.add('hidden');

                        // Inactive Styling (Reset to default)
                        if (isMobile) {
                            opt.classList.remove('text-blue-500', 'font-bold');
                            opt.classList.add('text-primary-40');
                        } else {
                            opt.classList.remove('bg-primary-70', 'text-white');
                            opt.classList.add('text-gray-300');
                        }
                    }
                });
            };

            updateIndicators(sleepTimerOptions, false);
            updateIndicators(sleepTimerOptionsMobile, true);
        }

        function clearSleepTimer() {
            if (sleepTimerId) {
                clearTimeout(sleepTimerId);
                sleepTimerId = null;
            }
            sleepMode = null;
            currentSleepMinutes = null;
            updateSleepTimerUI();
            console.log('Sleep timer cleared');
        }

        function setSleepTimer(minutes) {
            clearSleepTimer();
            sleepMode = 'time';
            currentSleepMinutes = minutes;
            updateSleepTimerUI();

            const ms = minutes * 60 * 1000;
            console.log(`Sleep timer set for ${minutes} minutes`);

            sleepTimerId = setTimeout(() => {
                pauseAudio();
                clearSleepTimer();
            }, ms);
        }

        function setEndOfTrackTimer() {
            clearSleepTimer();
            sleepMode = 'end';
            updateSleepTimerUI();
            console.log('Sleep timer set to End of Track');
        }

        function handleTimerSelection(val) {
            if (val === 'off') {
                clearSleepTimer();
            } else if (val === 'end') {
                setEndOfTrackTimer();
            } else {
                const minutes = parseInt(val);
                if (!isNaN(minutes)) {
                    setSleepTimer(minutes);
                }
            }
        }



        // Mobile Dropdown Listeners
        sleepTimerOptionsMobile.forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                handleTimerSelection(opt.dataset.value);
                // Close popup after selection
                if (audioSleepTimerPopup) {
                    audioSleepTimerPopup.classList.add('opacity-0', 'invisible');
                }
            });
        });

        // Mobile Trigger (Separate Popup)
        if (sleepTimerMobileBtn && audioSleepTimerPopup) {
            sleepTimerMobileBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                popupBehaviour(audioSleepTimerPopup);
            });

            // Close when clicking outside
            document.addEventListener('click', (e) => {
                if (!audioSleepTimerPopup.classList.contains('invisible')) {
                    if (!audioSleepTimerPopup.contains(e.target) && !sleepTimerMobileBtn.contains(e.target)) {
                        audioSleepTimerPopup.classList.add('opacity-0', 'invisible');
                    }
                }
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
                toggleButtonState(speakerMobileBtn, isMuted);
            }

            if (soundedBtn) soundedBtn.addEventListener('click', mutedSound);
            if (mutedBtn) mutedBtn.addEventListener('click', mutedSound);
            if (speakerMobileBtn) {
                speakerMobileBtn.addEventListener('click', (e) => {
                    mutedSound();
                });
            }

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
            if (sleepMode === 'end') {
                pauseAudio();
                clearSleepTimer();
                return;
            }

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

            function syncMobileMenuState() {
                toggleButtonState(loopMobileBtn, isLoop);
                toggleButtonState(shuffleMobileBtn, isShuffle);
                toggleButtonState(speakerMobileBtn, isMuted);
            }

            audioCtrlArea.addEventListener('click', (e) => {
                e.stopPropagation();
                if (audioCtrlPopup.classList.contains('invisible')) {
                    syncMobileMenuState();
                }
                popupBehaviour(audioCtrlPopup);
            });

            document.addEventListener('mousedown', (e) => {
                const sleepTimerMobileBtn = document.getElementById('sleep-timer-mobile');
                const audioSleepTimerPopup = document.getElementById('audioSleepTimerPopup');

                const isSleepTimerInteraction = (sleepTimerMobileBtn && sleepTimerMobileBtn.contains(e.target)) ||
                    (audioSleepTimerPopup && !audioSleepTimerPopup.classList.contains('invisible') && audioSleepTimerPopup.contains(e.target));

                if (audioCtrlContent && !audioCtrlContent.contains(e.target) && !audioCtrlArea.contains(e.target) && !isSleepTimerInteraction) {
                    audioCtrlPopup.classList.add('opacity-0', 'invisible');

                    // Also close sleep timer popup if main popup closes (optional, but cleaner)
                    if (audioSleepTimerPopup) {
                        audioSleepTimerPopup.classList.add('opacity-0', 'invisible');
                    }
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

            // Build initial play order and render tracks
            buildPlayOrder();
            renderPopupTracks();
            renderPopupLyrics();
        });

        // Listen for song changes to update media session
        window.addEventListener('song-changed', () => {
            updateMediaSessionMetadata();
        });

        // Listen for playlist updates (e.g., after navigating to a new playlist page)
        window.addEventListener('playlist-updated', () => {
            playlist = window.currentPlaylist || [];
            currentIndex = 0;
            isShuffle = false;
            if (shuffleBtn) toggleButtonState(shuffleBtn, false);
            buildPlayOrder();
            renderPopupTracks();
            renderPopupLyrics();
            if (playlist.length > 0) {
                loadSong(0);
            }
        });

        // Debugging events (optional)
        audio.addEventListener('error', (e) => {
            console.error('Audio error:', e);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const audio = document.getElementById('audio');
    const playBtn = document.getElementById('play');

    if (!audio || !playBtn) return;

    // =================== BEAT VISUALIZER & VINYL SPIN ===================
    (function (audio, playBtn) {
        const canvas = document.querySelector('#audioVisualizer');
        const vinylDisc = document.getElementById('vinylDisc');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        let audioCtx = null;
        let analyser = null;
        let bufferLength = 0;
        let dataArray = null;
        let sourceCreated = false;
        let animationId = null;

        function resizeCanvas() {
            if (canvas.clientWidth > 0 && canvas.clientHeight > 0) {
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;
            }
        }

        window.addEventListener('resize', resizeCanvas);

        // Resize when popup becomes visible
        const popupContainer = canvas.closest('[x-data]');
        if (popupContainer) {
            const observer = new MutationObserver(() => {
                if (!popupContainer.classList.contains('invisible')) {
                    setTimeout(resizeCanvas, 100);
                }
            });
            observer.observe(popupContainer, { attributes: true, attributeFilter: ['class'] });
        }

        if (window.ResizeObserver) {
            const ro = new ResizeObserver(() => resizeCanvas());
            ro.observe(canvas.parentElement);
        }

        // Vinyl spin control
        function setVinylSpin(playing) {
            if (!vinylDisc) return;
            if (playing) {
                vinylDisc.classList.remove('vinyl-paused');
                vinylDisc.classList.add('vinyl-playing');
            } else {
                vinylDisc.classList.remove('vinyl-playing');
                vinylDisc.classList.add('vinyl-paused');
            }
        }

        audio.addEventListener('play', () => setVinylSpin(true));
        audio.addEventListener('pause', () => setVinylSpin(false));
        audio.addEventListener('ended', () => setVinylSpin(false));

        // Draw circular beat visualization
        function drawVisualizer() {
            animationId = requestAnimationFrame(drawVisualizer);
            if (!analyser || canvas.width === 0 || canvas.height === 0) return;

            analyser.getByteFrequencyData(dataArray);
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const radius = Math.min(cx, cy);
            // Canvas is 210% of the vinyl size.
            // Radius = Canvas Radius.
            // Vinyl Radius = Canvas Radius / 2.1.
            const vinylRatio = 0.25;
            const innerRadius = radius * vinylRatio;
            const outerRadius = radius;
            const barCount = 80;

            for (let i = 0; i < barCount; i++) {
                const angle = (i / barCount) * Math.PI * 2 - Math.PI / 2;

                // Use only the first 60% of the frequency data (bass/mids) where most energy is
                // This prevents "dead" bars at the end of the spectrum
                const effectiveBufferLength = Math.floor(bufferLength * 0.6);
                const dataIndex = Math.floor(i * effectiveBufferLength / barCount);

                // Add a boost to higher frequencies (which are naturally quieter)
                // Linear boost from 1.0 (at i=0) to 2.5 (at i=max)
                const boost = 1 + (i / barCount) * 1.5;

                // Clamp value to 0-1 range after boost
                const value = Math.min(1.0, (dataArray[dataIndex] / 255) * boost);

                // Minimum height logic
                // Ensure there are no zero heights, and base is higher
                const effectiveValue = 0.15 + (value * 0.35);

                const barLength = (outerRadius - innerRadius) * effectiveValue;

                const x1 = cx + Math.cos(angle) * innerRadius;
                const y1 = cy + Math.sin(angle) * innerRadius;
                const x2 = cx + Math.cos(angle) * (innerRadius + barLength);
                const y2 = cy + Math.sin(angle) * (innerRadius + barLength);

                const hue = (i / barCount) * 60 + 180;
                const alpha = 1.0;

                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                // Calculate endpoints for gradient (radial direction)
                // Center of canvas is (centerX, centerY)
                // But we are drawing a line from (x1,y1) to (x2,y2)
                // We want the gradient to be along the line length
                const gradient = ctx.createLinearGradient(x1, y1, x2, y2);

                // Get color from canvas data attribute
                const baseColor = canvas.dataset.beatColor || '#ffffff'; // Fallback

                // create gradient: Base Color -> Transparent
                gradient.addColorStop(0, baseColor); // Inner part (intense)
                gradient.addColorStop(1, baseColor); // Outer part (solid)

                ctx.strokeStyle = gradient;

                // Wider bars
                ctx.lineWidth = (2 * Math.PI * innerRadius / barCount) - 1;
                ctx.lineCap = 'round';
                ctx.stroke();
            }
        }

        function initVisualizer() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 256;
                bufferLength = analyser.frequencyBinCount;
                dataArray = new Uint8Array(bufferLength);
            }

            if (!sourceCreated) {
                try {
                    const source = audioCtx.createMediaElementSource(audio);
                    source.connect(analyser);
                    analyser.connect(audioCtx.destination);
                    sourceCreated = true;
                } catch (e) {
                    console.warn('Visualizer source error:', e);
                }
            }

            if (audioCtx.state === 'suspended') audioCtx.resume();
            resizeCanvas();
            if (!animationId) drawVisualizer();
        }

        playBtn.addEventListener('click', initVisualizer, { once: true });

        audio.addEventListener('play', () => {
            if (!sourceCreated) initVisualizer();
            if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
        });

    })(audio, playBtn);
});
