if (!window.HAS_RUN_THREAD_JS) {
    window.HAS_RUN_THREAD_JS = true;

    // toggle comment
    // Use document listener which persists
    document.addEventListener("click", (e) => {
        // get and check the reply button
        const btn = e.target.closest('#toggleComment');
        if (!btn) return;

        // get the reply panels
        const targetId = btn.dataset.thread;
        const panel = document.getElementById(targetId);
        if (!panel) return;

        // show and hide the reply panel
        panel.classList.toggle('hidden');

        // make reply can be scrolled
        const replyContainer = panel.querySelector('.replyContainer');
        if (replyContainer) {
            replyContainer.scrollTop = replyContainer.scrollHeight;
        }
    });

    // pop-up function
    function popupBehaviour(element) {
        if (element.classList.contains('invisible')) element.classList.remove('opacity-0', 'invisible');
        else element.classList.add('opacity-0', 'invisible');
    }

    // Since `createThreadBtn` might be dynamic (in main content), we should probably use delegated events or re-attach on navigation.
    // But the original script used `const createThreadBtn = ...` at top level.
    // If we only run this once, `createThreadBtn` will be empty or stale if elements are replaced.
    // WE NEED TO RE-RUN ELEMENT SELECTION ON NAVIGATION.
    // So we should wrap the element selection and binding in a function `initThread` and call it on `navigated`.

    const initThread = () => {
        // create thread pop-up
        const createThreadBtn = document.querySelectorAll('.createThreadBtn');
        const createThreadPopup = document.getElementById('createThreadPopup');

        if (createThreadPopup) {
            const createThreadContent = createThreadPopup.querySelector('.popupContent');
            const closeCreateThreadBtn = createThreadPopup.querySelector('#closeCreateThread');

            // Remove old listeners? Hard to do without references.
            // But since we are selecting new elements (if re-rendered), we attach to new elements.
            // `createThreadPopup` might be outside main content (if in modal section).
            // If it matches `modal` component, it might vary.

            createThreadBtn.forEach(threadBtn => {
                // Check if listener attached? Custom property or just risk it (it's safe if elements are new).
                threadBtn.removeEventListener('click', threadBtn._clickHandler); // try remove if exists

                threadBtn._clickHandler = (e) => {
                    e.preventDefault();
                    popupBehaviour(createThreadPopup);
                };
                threadBtn.addEventListener('click', threadBtn._clickHandler);
            });

            if (closeCreateThreadBtn) {
                // This one might be persistent or re-rendered.
                // Better to use onclick in HTML or delegated event for safety, but let's just re-attach.
                // cloneNode(true) to strip listeners is a hack.
                // Simple idempotent way:
                closeCreateThreadBtn.onclick = (e) => {
                    e.preventDefault();
                    popupBehaviour(createThreadPopup);
                };
            }

            // close the popup when clicking outside - Document listener
            // We should NOT add this every time.
            // Move this OUTSIDE initThread or check existence.
        }
    };

    // Document listeners (run once)

    // close the popup when clicking outside
    document.addEventListener('mousedown', (e) => {
        const createThreadPopup = document.getElementById('createThreadPopup');
        if (createThreadPopup) {
            const createThreadContent = createThreadPopup.querySelector('.popupContent');
            if (createThreadContent && !createThreadContent.contains(e.target)) {
                createThreadPopup.classList.add('opacity-0', 'invisible');
            }
        }
    });

    // Validasi Scroll Reply (Livewire)
    document.addEventListener('livewire:init', () => {
        Livewire.on('reply-posted', (event) => {
            // 1. Ambil threadId dari event (Livewire 3 mengirim object)
            const threadId = event.threadId || event[0]?.threadId;

            console.log('Event received for Thread ID:', threadId);

            if (!threadId) return;

            // 2. Cari elemen panel reply
            const panel = document.getElementById('reply-' + threadId);

            if (panel) {
                // Pastikan panel terlihat
                panel.classList.remove('hidden');

                const replyContainer = panel.querySelector('.replyContainer');

                if (replyContainer) {
                    // 3. PENTING: Beri jeda waktu agar DOM baru selesai dirender
                    setTimeout(() => {
                        replyContainer.scrollTo({
                            top: replyContainer.scrollHeight,
                            behavior: 'smooth'
                        });

                        console.log('Scrolling executed. Height:', replyContainer.scrollHeight);
                    }, 250);
                }
            }
        });
    });

    // Run init logic
    document.addEventListener('DOMContentLoaded', initThread);
    document.addEventListener('livewire:navigated', initThread);
}