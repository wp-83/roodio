// toggle comment
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
    replyContainer.scrollTop = replyContainer.scrollHeight;
});

// pop-up function
function popupBehaviour(element){
    if(element.classList.contains('invisible')) element.classList.remove('opacity-0', 'invisible');
    else element.classList.add('opacity-0', 'invisible');
}

// create thread pop-up
const createThreadBtn = document.querySelectorAll('.createThreadBtn');
const createThreadPopup = document.getElementById('createThreadPopup');
const createThreadContent = createThreadPopup.querySelector('.popupContent');
const closeCreateThreadBtn = createThreadPopup.querySelector('#closeCreateThread');

createThreadBtn.forEach(threadBtn => {
    threadBtn.addEventListener('click', (e) => {
        e.preventDefault();
        popupBehaviour(createThreadPopup);
    })
});

closeCreateThreadBtn.addEventListener('click', (e) => {
    e.preventDefault();
    popupBehaviour(createThreadPopup);
});

// close the popup when clicking outside
document.addEventListener('mousedown', (e) => {
    if(!createThreadContent.contains(e.target)){
        createThreadPopup.classList.add('opacity-0', 'invisible');
    }
});

// Agar otomatis scroll ke bawah saat pesan baru terkirim
document.addEventListener('livewire:init', () => {
    Livewire.on('reply-posted', (event) => {
        // 1. Ambil threadId dari event (Livewire 3 mengirim object)
        // Kita gunakan destructuring agar aman, atau fallback ke event[0]
        const threadId = event.threadId || event[0]?.threadId;

        console.log('Event received for Thread ID:', threadId); // Cek Console browser

        if (!threadId) return;

        // 2. Cari elemen panel reply
        const panel = document.getElementById('reply-' + threadId);

        if (panel) {
            // Pastikan panel terlihat (jaga-jaga kalau tertutup)
            panel.classList.remove('hidden');

            const replyContainer = panel.querySelector('.replyContainer');
            
            if (replyContainer) {
                // 3. PENTING: Beri jeda waktu agar DOM baru selesai dirender
                // Kita gunakan setTimeout 250ms (cukup aman untuk mata manusia)
                setTimeout(() => {
                    // Pakai scrollTo dengan behavior smooth agar terlihat bergerak
                    replyContainer.scrollTo({
                        top: replyContainer.scrollHeight,
                        behavior: 'smooth'
                    });
                    
                    // Fallback: Paksa scroll instant jika smooth gagal (double strike)
                    // replyContainer.scrollTop = replyContainer.scrollHeight; 
                    
                    console.log('Scrolling executed. Height:', replyContainer.scrollHeight);
                }, 250); 
            }
        }
    });
});