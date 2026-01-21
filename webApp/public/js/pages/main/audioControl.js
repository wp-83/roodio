// take all audio play components
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
const overlayAudioPlay = player.querySelector('#overlayNan');   

// set audio path
let playlist = window.currentPlaylist || [];
let currentIndex = 0;

// load all songs in playlist
function loadSong(index) {
    if (playlist.length == 0) return;

    if (index < 0) index = 0;
    if (index >= playlist.length) index = playlist.length - 1;

    currentIndex = index;
    const song = playlist[currentIndex];
    audio.src = song.path;
    
    audio.load();
    if(!isSongNan()) overlayAudioPlay.classList.add('hidden');
};

// play audio
function playAudio() {
    if(isSongNan()) return;

    audio.play().then(() => {
        isPlay = true;
        playBtn.classList.add('hidden');
        pauseBtn.classList.remove('hidden');
    });
};

// pause audio
function pauseAudio() {
    audio.pause();
    isPlay = false;
    playBtn.classList.remove('hidden');
    pauseBtn.classList.add('hidden');
};

// toggle play
function togglePlay() {
    if (isPlay) {
        pauseAudio();
    } else {
        playAudio();
    }
};

// ensure the audio has been loaded
function playAfterLoad() {
    const handler = () => {
        playAudio();
        audio.removeEventListener('loadeddata', handler);
    };

    audio.addEventListener('loadeddata', handler);
};

// global function to play song based on the index
window.playByIndex = (index) => {
    loadSong(index);

    audio.onloadeddata = () => {
        playAudio();
    };
};

// Next Button
nextBtn.addEventListener('click', () => {
    loadSong(currentIndex + 1);
    playAfterLoad();
});

// Prev Button
prevBtn.addEventListener('click', () => {
    loadSong(currentIndex - 1);
    playAfterLoad();
});

// current player condition
let isPlay = false;
let isMuted = false;

// check if the song is NaN
function isSongNan(){
    return (audio.src == '') ? true : false;
}

//DOMContentLoaded Event
document.addEventListener('DOMContentLoaded', () => {
    if(playlist.length > 0) {
        loadSong(0); 
    }

    if(isSongNan()){
        overlayAudioPlay.classList.remove('hidden');
    } else {
        overlayAudioPlay.classList.add('hidden');
    }
});

// play and pause trigger
playBtn.addEventListener('click', togglePlay);
pauseBtn.addEventListener('click', togglePlay);

// keyboard shorcut for some audio behaviour
document.addEventListener('keydown', (e) => {
    // e.preventDefault();
    
    if(isSongNan() || audio.readyState < 2 || isNaN(audio.duration)) return;

    if(e.code == 'Space'){
        togglePlay();
    } 

    if (e.key == 'ArrowRight'){
        audio.currentTime += 10;
    }

    if (e.key == 'ArrowLeft'){
        audio.currentTime -= 10;
    }
});

// loop button behaviour
let loopActive = false;
loopBtn.addEventListener('click', (e) => {
    e.preventDefault();

    
});

// shuffle button behaviour
shuffleBtn.addEventListener('click', (e) => {
    e.preventDefault();

});

// intial value for volume slider
document.addEventListener('DOMContentLoaded', () => {
    volumeSlider.value = 1;
    audio.volume = volumeSlider.value;
});

// muted behaviour
let currVol = volumeSlider.value;

// muted sound behaviour
function mutedSound(){
    let tempVol = volumeSlider.value;

    if(!isMuted){
        isMuted = true;
        volumeSlider.value = 0;
        currVol = tempVol;
    } else {
        isMuted = false;
        volumeSlider.value = currVol;
    }
    
    audio.volume = volumeSlider.value;
    soundedBtn.classList.toggle('hidden');
    mutedBtn.classList.toggle('hidden');
}

soundedBtn.addEventListener('click', mutedSound);
mutedBtn.addEventListener('click', mutedSound);

// slider volume behaviour
volumeSlider.addEventListener('change', () => {
    if (volumeSlider.value == 0){
        isMuted = true;
        soundedBtn.classList.add('hidden');
        mutedBtn.classList.remove('hidden');
    } else if (soundedBtn.classList.contains('hidden')){
        mutedBtn.classList.add('hidden');
        soundedBtn.classList.remove('hidden');
        isMuted = false;
    }

    audio.volume = volumeSlider.value;
});

// get the element of progress bar player
const progressBar = document.getElementById("progressBar");
const currentTimeEl = document.getElementById("currentDuration");
const durationEl = document.getElementById("duration");

// function to format time
function formatTime(time) {
    const minutes = Math.floor(time / 60).toString().padStart(2, '0');
    const seconds = Math.floor(time % 60).toString().padStart(2, "0");
    return `${minutes}:${seconds}`;
}

// format time
audio.addEventListener("loadedmetadata", () => {
    durationEl.textContent = formatTime(audio.duration);
});

audio.addEventListener("timeupdate", () => {
    const percent = (audio.currentTime / audio.duration) * 100;
    progressBar.style.width = `${percent}%`;
    currentTimeEl.textContent = formatTime(audio.currentTime);
});

audio.addEventListener("ended", () => {
    pauseBtn.classList.add('hidden');
    playBtn.classList.remove('hidden');
    isPlay = false;

    if (currentIndex < playlist.length - 1) {
        playByIndex(currentIndex + 1);
    };
});

progressContainer.addEventListener("click", (e) => {
    const width = progressContainer.clientWidth;
    const clickX = e.offsetX;

    audio.currentTime = (clickX / width) * audio.duration;
});

// popup behaviour
function popupBehaviour(element){
    if(element.classList.contains('invisible')) element.classList.remove('opacity-0', 'invisible');
    else element.classList.add('opacity', 'invisible');
}

//audio control pop-up behaviour
const audioCtrlArea = document.getElementById('audioControlResponsive');
const audioCtrlPopup = document.getElementById('audioControlPopup');
const audioCtrlContent = audioCtrlPopup.querySelector('.popupContent');

// event trigger for mood
audioCtrlArea.addEventListener('click', () => {
    popupBehaviour(audioCtrlPopup);
    popupBehaviour(audioCtrlContent);
});

// close pop up when clicking outside
document.addEventListener('mousedown', (e) => {
    if(!audioCtrlContent.contains(e.target)){
        audioCtrlContent.classList.add('opacity-0', 'invisible');
        audioCtrlPopup.classList.add('opacity-0', 'invisible');
    }
}); 