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
