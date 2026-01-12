// shortcut for search bar in navbar
document.addEventListener('keydown', (e) => {
    // Check for Ctrl key (or Cmd key on macOS) and the K key
    if ((e.ctrlKey || e.metaKey) && (e.key === 'K' || e.key === 'k')) {
        // Prevent the default browser behavior for Ctrl+K (usually bookmarking)
        e.preventDefault();

        // Get the search input element by its ID
        const searchInput = document.getElementById('search');

        // If the input element exists, focus it
        if (searchInput) {
            searchInput.focus();
        }
    }

    // Optional: Add an 'Escape' key listener to blur the input
    if (e.key === 'Escape') {
        const searchInput = document.getElementById('search');
        if (searchInput) {
            searchInput.blur();
        }
    }
});

// remove input while the close icon click
const deleteSearchIcon = document.getElementById('searchClose');
deleteSearchIcon.addEventListener('mousedown', (e) => {
    // prevent default browser behaviour
    e.preventDefault();

    // set search bar value into empty
    const searchBar = document.getElementById('search');
    searchBar.value = '';

    // make the search bar keep focus
    setTimeout(() => {
        searchBar.focus();
    }, 0);
});