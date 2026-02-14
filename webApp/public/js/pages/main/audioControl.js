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

        // load all songs in playlist
        function loadSong(index) {
            if (playlist.length == 0) return;

            if (index < 0) index = 0;
            if (index >= playlist.length) index = playlist.length - 1;

            currentIndex = index;
            const song = playlist[currentIndex];
            audio.src = song.path;

            // Sync Metadata
            if (playerTitle) playerTitle.textContent = song.title || 'Unknown Title';
            if (playerArtist) playerArtist.textContent = song.artist || 'Unknown Artist';
            if (playerImage) playerImage.src = song.image || '';

            // Dispatch event for Alpine.js popup
            window.dispatchEvent(new CustomEvent('song-changed', {
                detail: {
                    title: song.title || 'Unknown Title',
                    artist: song.artist || 'Unknown Artist',
                    image: song.image || ''
                }
            }));

            audio.load();
            if (!isSongNan() && overlayAudioPlay) overlayAudioPlay.classList.add('hidden');

            updatePlaylistVisuals(index);
        };

        // Update Playlist Highlight
        function updatePlaylistVisuals(index) {
            // Remove active style from all
            // We can't easily query all *potential* buttons if list is huge, but we can query by class or assume IDs.
            // Since we use ID "song-X", let's loop or querySelectorAll.
            const allSongs = document.querySelectorAll(`[id^="song-"]`);
            allSongs.forEach(btn => {
                const activeClass = btn.dataset.activeClass;
                const innerDiv = btn.firstElementChild; // The div inside the button has the bg
                if (activeClass && innerDiv) {
                    activeClass.split(' ').forEach(c => c && innerDiv.classList.remove(c));
                }
            });

            // Add active style to current
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
            // Play promise to avoid interruption errors
            const playPromise = audio.play();
            if (playPromise !== undefined) {
                playPromise.then(() => {
                    isPlay = true;
                    playBtn.classList.add('hidden');
                    pauseBtn.classList.remove('hidden');

                    // Auto-open popup on first play
                    if (!hasPlayedOnce) {
                        hasPlayedOnce = true;
                        window.dispatchEvent(new CustomEvent('open-player-popup'));
                    }
                }).catch(e => console.error("Play error:", e));
            }
        };

        function pauseAudio() {
            audio.pause();
            isPlay = false;
            playBtn.classList.remove('hidden');
            pauseBtn.classList.add('hidden');
        };

        function togglePlay() {
            if (isPlay) {
                pauseAudio();
            } else {
                playAudio();
            }
        };

        function playAfterLoad() {
            const handler = () => {
                playAudio();
                audio.removeEventListener('loadeddata', handler);
            };
            audio.addEventListener('loadeddata', handler);
        };

        window.playByIndex = (index) => {
            if (window.currentPlaylist) {
                playlist = window.currentPlaylist;
            }
            loadSong(index);
            audio.onloadeddata = () => {
                playAudio();
            };
        };

        function getRandomIndex() {
            if (playlist.length <= 1) return 0;
            let newIndex;
            do {
                newIndex = Math.floor(Math.random() * playlist.length);
            } while (newIndex === currentIndex);
            return newIndex;
        }

        // NEXT
        function playNext() {
            if (isShuffle) {
                loadSong(getRandomIndex());
            } else {
                let nextIndex = currentIndex + 1;
                if (nextIndex >= playlist.length) nextIndex = 0;
                loadSong(nextIndex);
            }
            playAfterLoad();
        }

        if (nextBtn) {
            nextBtn.addEventListener('click', playNext);
        }

        // PREV
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                let prevIndex = currentIndex - 1;
                if (prevIndex < 0) prevIndex = playlist.length - 1;
                loadSong(prevIndex);
                playAfterLoad();
            });
        }

        // VISUAL STATE
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

        function isSongNan() {
            return (audio.src == '' || audio.src.endsWith('undefined')) ? true : false;
        }

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
        });

        if (playBtn) playBtn.addEventListener('click', togglePlay);
        if (pauseBtn) pauseBtn.addEventListener('click', togglePlay);

        // KEYBOARD SHORTCUTS
        document.addEventListener('keydown', (e) => {
            // Guard clauses: Must have audio source, ready state, and finite duration
            if (isSongNan() || audio.readyState < 1 || !Number.isFinite(audio.duration)) return;

            if (e.code == 'Space') {
                e.preventDefault();
                togglePlay();
            }

            if (e.key == 'ArrowRight') {
                e.preventDefault(); // Prevent scrolling
                // Move forward 10s
                let newTime = audio.currentTime + 10;
                // Clamp to duration - 0.1 to avoid ending
                if (newTime > audio.duration) newTime = audio.duration - 0.1;
                if (Number.isFinite(newTime)) audio.currentTime = newTime;
            }

            if (e.key == 'ArrowLeft') {
                e.preventDefault(); // Prevent scrolling
                // Move backward 10s
                let newTime = audio.currentTime - 10;
                if (newTime < 0) newTime = 0;
                if (Number.isFinite(newTime)) audio.currentTime = newTime;
            }
        });

        // VOLUME
        if (volumeSlider) {
            document.addEventListener('DOMContentLoaded', () => {
                volumeSlider.value = 1;
                audio.volume = volumeSlider.value;
            });

            let currVol = volumeSlider.value;

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

            volumeSlider.addEventListener('change', () => {
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
        }

        const progressBar = document.getElementById("progressBar");
        const currentTimeEl = document.getElementById("currentDuration");
        const durationEl = document.getElementById("duration");
        const progressContainer = document.getElementById('progressContainer');

        function formatTime(time) {
            if (isNaN(time) || !Number.isFinite(time)) return "00:00";
            const minutes = Math.floor(time / 60).toString().padStart(2, '0');
            const seconds = Math.floor(time % 60).toString().padStart(2, "0");
            return `${minutes}:${seconds}`;
        }

        if (progressBar && currentTimeEl && durationEl) {
            audio.addEventListener("loadedmetadata", () => {
                if (Number.isFinite(audio.duration)) {
                    durationEl.textContent = formatTime(audio.duration);
                }
            });

            audio.addEventListener("timeupdate", () => {
                if (Number.isFinite(audio.duration)) {
                    const percent = (audio.currentTime / audio.duration) * 100;
                    progressBar.style.width = `${percent}%`;
                    currentTimeEl.textContent = formatTime(audio.currentTime);
                }
            });
        }

        // ENDED Listener
        audio.addEventListener("ended", () => {
            if (!audio.loop) {
                playNext();
                if (!isPlay) {
                    pauseBtn.classList.add('hidden');
                    playBtn.classList.remove('hidden');
                }
            }
        });

        // SEEK CLICK
        if (progressContainer) {
            progressContainer.addEventListener("click", (e) => {
                // STRICT CHECK: Duration must be finite and > 0
                if (!audio.duration || !Number.isFinite(audio.duration) || audio.duration <= 0) return;

                const rect = progressContainer.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const width = rect.width;

                if (width > 0) {
                    const percentage = clickX / width;
                    // Clamp percentage 0-1
                    const clampedPercentage = Math.max(0, Math.min(1, percentage));
                    const newTime = clampedPercentage * audio.duration;

                    if (Number.isFinite(newTime)) {
                        audio.currentTime = newTime;
                    }
                }
            });
        }

        function popupBehaviour(element) {
            if (element.classList.contains('invisible')) element.classList.remove('opacity-0', 'invisible');
            else element.classList.add('opacity-0', 'invisible');
        }

        const audioCtrlArea = document.getElementById('audioControlResponsive');
        const audioCtrlPopup = document.getElementById('audioControlPopup');

        if (audioCtrlArea && audioCtrlPopup) {
            const audioCtrlContent = audioCtrlPopup.querySelector('.popupContent');

            audioCtrlArea.addEventListener('click', () => {
                popupBehaviour(audioCtrlPopup);
            });

            document.addEventListener('mousedown', (e) => {
                if (!audioCtrlContent.contains(e.target) && !audioCtrlArea.contains(e.target)) {
                    audioCtrlContent.classList.add('opacity-0', 'invisible');
                    audioCtrlPopup.classList.add('opacity-0', 'invisible');
                }
            });
        }
    }
}