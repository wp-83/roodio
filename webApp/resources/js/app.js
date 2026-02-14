import './bootstrap';
import 'flowbite';

// fade effect on the elements
const initFadeEffect = () => {
    const container = document.querySelector('#scrollContainer');
    const elements = document.querySelectorAll('.contentFadeLoad');

    // if container doesn't exist
    if (!container) return;

    function checkElements() {
        elements.forEach(element => {

            const elementOffset = element.offsetTop; //container height
            const containerScroll = container.scrollTop; //container top position
            const containerHeight = container.clientHeight - 25; //container that viewed - tolerance margin

            const relativePosition = elementOffset - containerScroll; // element position based on top position of container

            if (relativePosition < containerHeight) {
                element.classList.add('show');
            }
        });
    }

    // scroll trigger
    // Remove previous listener to avoid duplicates if possible, OR just add.
    // Since initFadeEffect runs on navigation, new container = new listener.
    // Old container is gone. So simple addEventListener is fine.
    container.addEventListener('scroll', checkElements);

    // first trigger
    checkElements();
};

document.addEventListener('DOMContentLoaded', initFadeEffect);
document.addEventListener('livewire:navigated', initFadeEffect);