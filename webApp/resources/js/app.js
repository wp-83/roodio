import './bootstrap';
import 'flowbite';

// fade animation when elements are loaded
window.addEventListener("load", () => {
    const elements = document.querySelectorAll(".contentFadeLoad");

    if (elements.length > 0) {
        elements.forEach(element => element.classList.add("show"));
    }
});