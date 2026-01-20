const replyContainer = document.querySelectorAll('#replyContainer');

// replyContainer.scrollTop = replyContainer.scrollHeight;

replyContainer.forEach((reply) => {
    reply.scrollTop = reply.scrollHeight;
    
});