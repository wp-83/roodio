const searchInput = document.getElementById('search');
const searchbar = document.getElementById('searchbar');

if (searchInput && searchbar) {
    // shortcut for search bar in navbar
    document.addEventListener('keydown', (e) => {
        // Check for Ctrl key (or Cmd key on macOS) and the K key
        if ((e.ctrlKey || e.metaKey) && (e.key === 'K' || e.key === 'k')) {
            // Prevent the default browser behavior for Ctrl+K (usually bookmarking)
            e.preventDefault();
            searchInput.focus();
        }

        // Optional: Add an 'Escape' key listener to blur the input
        if (e.key === 'Escape') {
            searchInput.blur();
        }
    });

    // remove input while the close icon click
    const deleteSearchIcon = document.getElementById('searchClose');
    if (deleteSearchIcon) {
        deleteSearchIcon.addEventListener('mousedown', (e) => {
            // prevent default browser behaviour
            e.preventDefault();

            // set search bar value into empty
            searchInput.value = '';

            // make the search bar keep focus
            setTimeout(() => {
                searchInput.focus();
            }, 0);
        });
    }

    // search bar icon behaviour
    const searchIcon = document.getElementById('searchIcon');
    const searchAttr = searchbar.querySelector('#searchContent');
    let isSearchBarOpened = false;

    // open the search bar responsive
    function openSearchBar() {
        isSearchBarOpened = true;
        searchbar.classList.add("absolute", 'z-10', 'top-[10%]', 'left-1/2', '-translate-x-1/2', 'w-sm');
        if (searchAttr) searchAttr.classList.add('hidden');
        searchbar.classList.remove('hidden', 'relative', 'w-xl', 'h-max');

        searchInput.focus();
    };

    // close the search bar responsive
    function closeSearchBar() {
        isSearchBarOpened = false;
        searchbar.classList.add('-z-1');
        searchbar.classList.add('hidden', 'w-xl', 'relative', 'h-max');
        searchbar.classList.remove('z-10', 'top-[10%]', 'left-1/2', '-translate-x-1/2', 'w-sm');
        searchbar.classList.remove('absolute', '-z-1');
        if (searchAttr) searchAttr.classList.remove('hidden');
    };

    // search bar responsive trigger
    if (searchIcon) {
        searchIcon.addEventListener('click', () => {
            (isSearchBarOpened) ? closeSearchBar() : openSearchBar();
        });
    }

    // document.addEventListener('mousedown')
    document.addEventListener('mousedown', (e) => {
        const isClickInsideSearch = (searchbar.contains(e.target) || (searchIcon && searchIcon.contains(e.target)));

        if (!isClickInsideSearch && isSearchBarOpened) {
            closeSearchBar();
        }
    });

    // back to the default style of searchbar after responsive behaviour
    window.addEventListener('resize', () => {
        if (isSearchBarOpened && window.innerWidth >= 768) closeSearchBar();
    });
}


// sidebar elements
const hamburgerBtn = document.getElementById('hamburgerBtn');
const hamburgerLines = hamburgerBtn.querySelectorAll('#hamburgerLine');
const musicNote = hamburgerBtn.querySelector('#musicNote');

const sidebar = document.getElementById('sidebar');
const notToggleSidebarBtns = sidebar.querySelectorAll('#notToggleSidebar');
const toggleSidebarBtns = sidebar.querySelectorAll('#toggleSidebar');

let isSidebarOpen = false;
const MOBILE_WIDTH = 768;

// initial animation
document.addEventListener('DOMContentLoaded', () => {
    musicNote.classList.add('invisible');
    handleResponsive();
});

// open sidebar function
function openSidebar() {
    isSidebarOpen = true;

    // hamburger animation
    hamburgerLines.forEach(h => {
        h.classList.add('expandWidth');
        h.classList.remove('collapseWidth');
    });

    // music note
    musicNote.classList.remove('invisible');
    musicNote.classList.add('bouncyNote');

    // sidebar content
    toggleSidebarBtns.forEach(btn => btn.classList.remove('hidden'));
    notToggleSidebarBtns.forEach(btn => btn.classList.add('hidden'));

    sidebar.classList.add('pt-5', 'px-3');

    // mobile behavior
    if (isMobile()) {
        sidebar.classList.add('translate-x-0', 'z-5');
        sidebar.classList.remove('-translate-x-full');
    }
}

// close sidebar function
function closeSidebar() {
    isSidebarOpen = false;

    // hamburger animation
    hamburgerLines.forEach(h => {
        h.classList.add('collapseWidth');
        h.classList.remove('expandWidth');
    });

    // music note
    musicNote.classList.add('invisible');
    musicNote.classList.remove('bouncyNote');

    // sidebar content
    toggleSidebarBtns.forEach(btn => btn.classList.add('hidden'));
    notToggleSidebarBtns.forEach(btn => btn.classList.remove('hidden'));

    sidebar.classList.remove('pt-5', 'px-3');

    // mobile behavior
    if (isMobile()) {
        sidebar.classList.add('-translate-x-full');
        sidebar.classList.remove('translate-x-0', 'z-5');
    }
}

// helper for checking window width size
function isMobile() {
    return window.innerWidth < MOBILE_WIDTH;
}

// responsive behaviour
function handleResponsive() {
    if (isMobile()) {
        sidebar.classList.add('absolute');
        if (sidebar.classList.contains('relative')) sidebar.classList.remove('relative');

        if (!isSidebarOpen) {
            sidebar.classList.add('-translate-x-full');
            sidebar.classList.remove('translate-x-0', 'z-5');
        }
    } else {
        sidebar.classList.remove('absolute', '-translate-x-full');
        sidebar.classList.add('translate-x-0');
    }
}

// trigger function
hamburgerBtn.addEventListener('click', () => {
    isSidebarOpen ? closeSidebar() : openSidebar();
});

window.addEventListener('resize', handleResponsive);

// popup behaviour
function popupBehaviour(element) {
    if (element.classList.contains('invisible')) element.classList.remove('opacity-0', 'invisible');
    else element.classList.add('opacity-0', 'invisible');
}

// profile pop-up behaviour
const profileArea = document.getElementById('profileNavbar');
const profilePopup = document.getElementById('profilePopup');
const profileContent = profilePopup.querySelector('.popupContent');

// event trigger for profile
profileArea.addEventListener('click', () => {
    popupBehaviour(profilePopup);
});

// close icon behaviour on profile popup
profilePopup.querySelector('#closeProfilePopup').addEventListener('click', () => {
    profilePopup.classList.add('opacity-0', 'invisible');
});

//mood pop-up behaviour
const moodArea = document.getElementById('currentMood');
const moodPopup = document.getElementById('changeMood');
const moodContent = moodPopup.querySelector('.popupContent');

// event trigger for mood
moodArea.addEventListener('click', () => {
    popupBehaviour(moodPopup);
});

// close popups on outside click
document.addEventListener('mousedown', (e) => {
    // profile popup close
    if (!profileContent.contains(e.target)) {
        profilePopup.classList.add('opacity-0', 'invisible');
    }

    // mood popup close
    if (!moodContent.contains(e.target)) {
        moodPopup.classList.add('opacity-0', 'invisible');
    }
});