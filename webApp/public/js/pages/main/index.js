document.addEventListener('DOMContentLoaded', () => {

    const firstPopup = document.getElementById('dayMood');
    const secondPopup = document.getElementById('choosePlaylist');

    // show first pop-up
    if (firstPopup) {
        firstPopup.classList.remove('opacity-0', 'invisible');
    }

    if (secondPopup) {
        // show second pop-up
        secondPopup.classList.remove('opacity-0', 'invisible');
    }

});
