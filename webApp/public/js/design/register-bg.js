document.addEventListener('DOMContentLoaded', function () {
    const canvasDiv = document.getElementById('particle-canvas');

    if (!canvasDiv) {
        console.error('particle-canvas not found');
        return;
    }

    canvasDiv.style.width = '100%';

    const options = {
        particleColor: ['#ffffff'],
        background: 'transparent',
        interactive: true,
        speed: 'fast',
        density: 'high',
    };

    new ParticleNetwork(canvasDiv, options);
});