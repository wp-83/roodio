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
    const replyContainer = document.querySelector('.replyContainer');
    replyContainer.scrollTop = replyContainer.scrollHeight;
});