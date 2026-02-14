if (!window.HAS_RUN_NAVIGATION_JS) {
    window.HAS_RUN_NAVIGATION_JS = true;

    const initNavigation = () => {
        const searchInput = document.getElementById('search');
        const searchbar = document.getElementById('searchbar');

        if (searchInput && searchbar) {
            // shortcut for search bar in navbar
            // Use a named function to avoid duplicate listeners if possible, or just accept it for now as low risk
            document.addEventListener('keydown', (e) => {
                if ((e.ctrlKey || e.metaKey) && (e.key === 'K' || e.key === 'k')) {
                    e.preventDefault();
                    searchInput.focus();
                }
                if (e.key === 'Escape') {
                    searchInput.blur();
                }
            });

            const deleteSearchIcon = document.getElementById('searchClose');
            if (deleteSearchIcon) {
                deleteSearchIcon.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    searchInput.value = '';
                    setTimeout(() => {
                        searchInput.focus();
                    }, 0);
                });
            }

            const searchIcon = document.getElementById('searchIcon');
            const searchAttr = searchbar.querySelector('#searchContent');
            let isSearchBarOpened = false;

            function openSearchBar() {
                isSearchBarOpened = true;
                searchbar.classList.add("absolute", 'z-10', 'top-[10%]', 'left-1/2', '-translate-x-1/2', 'w-sm');
                if (searchAttr) searchAttr.classList.add('hidden');
                searchbar.classList.remove('hidden', 'relative', 'w-xl', 'h-max');
                searchInput.focus();
            };

            function closeSearchBar() {
                isSearchBarOpened = false;
                searchbar.classList.add('-z-1');
                searchbar.classList.add('hidden', 'w-xl', 'relative', 'h-max');
                searchbar.classList.remove('z-10', 'top-[10%]', 'left-1/2', '-translate-x-1/2', 'w-sm');
                searchbar.classList.remove('absolute', '-z-1');
                if (searchAttr) searchAttr.classList.remove('hidden');
            };

            if (searchIcon) {
                searchIcon.addEventListener('click', () => {
                    (isSearchBarOpened) ? closeSearchBar() : openSearchBar();
                });
            }

            document.addEventListener('mousedown', (e) => {
                const isClickInsideSearch = (searchbar.contains(e.target) || (searchIcon && searchIcon.contains(e.target)));
                if (!isClickInsideSearch && isSearchBarOpened) {
                    closeSearchBar();
                }
            });

            window.addEventListener('resize', () => {
                if (isSearchBarOpened && window.innerWidth >= 768) closeSearchBar();
            });
        }

        // sidebar elements
        const hamburgerBtn = document.getElementById('hamburgerBtn');

        // Only proceed if elements exist
        if (hamburgerBtn) {
            const hamburgerLines = hamburgerBtn.querySelectorAll('#hamburgerLine');
            const musicNote = hamburgerBtn.querySelector('#musicNote');
            const sidebar = document.getElementById('sidebar');

            if (sidebar) {
                const notToggleSidebarBtns = sidebar.querySelectorAll('#notToggleSidebar');
                const toggleSidebarBtns = sidebar.querySelectorAll('#toggleSidebar');

                let isSidebarOpen = false;
                const MOBILE_WIDTH = 768;

                // initial animation
                musicNote.classList.add('invisible');
                handleResponsive();

                function openSidebar() {
                    isSidebarOpen = true;
                    hamburgerLines.forEach(h => {
                        h.classList.add('expandWidth');
                        h.classList.remove('collapseWidth');
                    });
                    musicNote.classList.remove('invisible');
                    musicNote.classList.add('bouncyNote');
                    toggleSidebarBtns.forEach(btn => btn.classList.remove('hidden'));
                    notToggleSidebarBtns.forEach(btn => btn.classList.add('hidden'));
                    sidebar.classList.add('pt-5', 'px-3');

                    if (isMobile()) {
                        sidebar.classList.add('translate-x-0', 'z-5');
                        sidebar.classList.remove('-translate-x-full');
                    }
                }

                function closeSidebar() {
                    isSidebarOpen = false;
                    hamburgerLines.forEach(h => {
                        h.classList.add('collapseWidth');
                        h.classList.remove('expandWidth');
                    });
                    musicNote.classList.add('invisible');
                    musicNote.classList.remove('bouncyNote');
                    toggleSidebarBtns.forEach(btn => btn.classList.add('hidden'));
                    notToggleSidebarBtns.forEach(btn => btn.classList.remove('hidden'));
                    sidebar.classList.remove('pt-5', 'px-3');

                    if (isMobile()) {
                        sidebar.classList.add('-translate-x-full');
                        sidebar.classList.remove('translate-x-0', 'z-5');
                    }
                }

                function isMobile() {
                    return window.innerWidth < MOBILE_WIDTH;
                }

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

                hamburgerBtn.addEventListener('click', () => {
                    isSidebarOpen ? closeSidebar() : openSidebar();
                });

                window.addEventListener('resize', handleResponsive);
            }
        }

        // Popup behavior 
        // We need to re-attach these listeners as the elements might be re-rendered
        const popupBehaviour = (element) => {
            if (element.classList.contains('invisible')) element.classList.remove('opacity-0', 'invisible');
            else element.classList.add('opacity-0', 'invisible');
        }

        const profileArea = document.getElementById('profileNavbar');
        if (profileArea) {
            const profilePopup = document.getElementById('profilePopup');
            const profileContent = profilePopup.querySelector('.popupContent');

            profileArea.addEventListener('click', () => {
                popupBehaviour(profilePopup);
            });

            const closeProfileBtn = profilePopup.querySelector('#closeProfilePopup');
            if (closeProfileBtn) {
                closeProfileBtn.addEventListener('click', () => {
                    profilePopup.classList.add('opacity-0', 'invisible');
                });
            }

            // Store references to remove listeners later if needed, or rely on garbage collection of replaced nodes
            // For global document listeners, we should be careful.
            // A simple way for now:
            document.addEventListener('mousedown', (e) => {
                if (!profileContent.contains(e.target) && !profilePopup.classList.contains('invisible')) {
                    profilePopup.classList.add('opacity-0', 'invisible');
                }
            });
        }

        const moodArea = document.getElementById('currentMood');
        if (moodArea) {
            const moodPopup = document.getElementById('changeMood');
            const moodContent = moodPopup.querySelector('.popupContent');

            moodArea.addEventListener('click', () => {
                popupBehaviour(moodPopup);
            });

            // Combined document listener for popups could be optimized, but separate functional blocks are fine
            document.addEventListener('mousedown', (e) => {
                if (!moodContent.contains(e.target) && !moodPopup.classList.contains('invisible')) {
                    moodPopup.classList.add('opacity-0', 'invisible');
                }
            });
        }
    };

    // Initialize on load
    document.addEventListener('DOMContentLoaded', initNavigation);
    // Initialize on Livewire navigation
    document.addEventListener('livewire:navigated', initNavigation);
}