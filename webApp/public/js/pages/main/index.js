const initIndex = () => {
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

    // Fade in content
    const fadeElements = document.querySelectorAll('.contentFadeLoad');
    fadeElements.forEach(el => {
        // slight delay to ensure transition works if added immediately after dom update
        setTimeout(() => {
            el.classList.add('show');
        }, 50);
    });
};

document.addEventListener('DOMContentLoaded', initIndex);
document.addEventListener('livewire:navigated', initIndex);
