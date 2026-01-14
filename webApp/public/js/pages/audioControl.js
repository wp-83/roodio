// take all audio play components
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

// set audio path
audio.src = 'https://roodio.blob.core.windows.net/uploads/songs/dKuze1CQQwO0cU7jLP3ZCvxwX13uaDyNKpL15zBi.mp3';

// current player condition
let isPlay = false;
let isMuted = false;

// main logic play and pause condition
function playMusic(){
    if (!isPlay){
        isPlay = true;
        audio.play();
    } else {
        isPlay = false;
        audio.pause();
    }

    // toggle class display
    playBtn.classList.toggle('hidden');
    pauseBtn.classList.toggle('hidden');
};

// play and pause trigger
playBtn.addEventListener('click', playMusic);
pauseBtn.addEventListener('click', playMusic);

// loop button behaviour
loopBtn.addEventListener('click', (e) => {
    e.preventDefault();


});

// shuffle button behaviour
shuffleBtn.addEventListener('click', (e) => {

});

// soundedBtn, mutedBtn, volumeSlider

// intial value for volume slider
document.addEventListener('DOMContentLoaded', () => {
    volumeSlider.value = 1;
    audio.volume = volumeSlider.value;
});

// muted behaviour
let currVol = volumeSlider.value;

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

audio.addEventListener("loadedmetadata", () => {
    durationEl.textContent = formatTime(audio.duration);
});

audio.addEventListener("timeupdate", () => {
    const percent = (audio.currentTime / audio.duration) * 100;
    progressBar.style.width = `${percent}%`;
    currentTimeEl.textContent = formatTime(audio.currentTime);

    if (audio.currentTime == audio.duration){
        pauseBtn.classList.add('hidden');
        playBtn.classList.remove('hidden');
        isPlay = false;
    };  
});

progressContainer.addEventListener("click", (e) => {
    const width = progressContainer.clientWidth;
    const clickX = e.offsetX;

    audio.currentTime =
        (clickX / width) * audio.duration;
});
