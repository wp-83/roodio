document.addEventListener('keydown', function(event) {
    // Check for Ctrl key (or Cmd key on macOS) and the K key
    if ((event.ctrlKey || event.metaKey) && event.key === 'K') {
        // Prevent the default browser behavior for Ctrl+K (usually bookmarking)
        event.preventDefault();

        // Get the search input element by its ID
        const searchInput = document.getElementById('search-input');

        // If the input element exists, focus it
        if (searchInput) {
            searchInput.focus();
        }
    }

    // Optional: Add an 'Escape' key listener to blur the input
    if (event.key === 'Escape') {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.blur();
        }
    }
});
