import './bootstrap';
import 'flowbite';

// fade effect on the elements
document.addEventListener('DOMContentLoaded', () => {
    const container = document.querySelector('#scrollContainer');
    const elements = document.querySelectorAll('.contentFadeLoad');

    // if container doesn't exist
    if (!container) return;

    function checkElements() {
        elements.forEach(element => {

            const elementOffset = element.offsetTop; //container height
            const containerScroll = container.scrollTop; //container top position
            const containerHeight = container.clientHeight -25; //container that viewed - tolerance margin

            const relativePosition = elementOffset - containerScroll; // element position based on top position of container

            if (relativePosition < containerHeight) {
                element.classList.add('show');
            } else {
                element.classList.remove('show');
            }
        });
    }

    // scroll trigger
    container.addEventListener('scroll', checkElements);

    // first trigger
    checkElements();
});