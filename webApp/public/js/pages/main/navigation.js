if (!window.HAS_RUN_NAVIGATION_JS) {
    window.HAS_RUN_NAVIGATION_JS = true;

    // Helper functions (Dynamic Lookup)
    function getElement(id) {
        return document.getElementById(id);
    }

    function togglePopup(popupId) {
        const popup = getElement(popupId);
        if (!popup) return;

        if (popup.classList.contains('invisible')) {
            popup.classList.remove('opacity-0', 'invisible');
        } else {
            popup.classList.add('opacity-0', 'invisible');
        }
    }

    function closePopup(popupId) {
        const popup = getElement(popupId);
        if (popup) {
            popup.classList.add('opacity-0', 'invisible');
        }
    }

    const initNavigation = () => {
        const searchInput = document.getElementById('search');
        const searchbar = document.getElementById('searchbar');

        if (searchInput && searchbar) {
            // shortcut for search bar in navbar
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

            // ==========================================
            // SEARCH BAR LOGIC (Original Implementation)
            // ==========================================
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

            // Direct listener (User Preference)
            if (searchIcon) {
                // Remove old listener if any (to prevent duplicates if init runs twice on same element)
                // Note: Named functions are better for removal, but here we accept potential accumulation 
                // if init runs multiple times on same element, though 'livewire:navigated' usually means new DOM.
                // Ideally we'd clone or use named function. For now, simple standard addEventListener.
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
                // Check innerWidth to see if we are in desktop or mobile
                if (isSearchBarOpened && window.innerWidth >= 768) closeSearchBar();
            });
        }

        // sidebar elements
        const hamburgerBtn = document.getElementById('hamburgerBtn');
        if (hamburgerBtn) {
            const hamburgerLines = hamburgerBtn.querySelectorAll('#hamburgerLine');
            const musicNote = hamburgerBtn.querySelector('#musicNote');
            const sidebar = document.getElementById('sidebar');

            if (sidebar) {
                const notToggleSidebarBtns = sidebar.querySelectorAll('#notToggleSidebar');
                const toggleSidebarBtns = sidebar.querySelectorAll('#toggleSidebar');
                let isSidebarOpen = false;
                const MOBILE_WIDTH = 768;

                musicNote.classList.add('invisible');
                handleResponsive();

                function openSidebar() {
                    isSidebarOpen = true;
                    hamburgerLines.forEach(h => { h.classList.add('expandWidth'); h.classList.remove('collapseWidth'); });
                    musicNote.classList.remove('invisible'); musicNote.classList.add('bouncyNote');
                    toggleSidebarBtns.forEach(btn => btn.classList.remove('hidden'));
                    notToggleSidebarBtns.forEach(btn => btn.classList.add('hidden'));
                    sidebar.classList.add('pt-5', 'px-3');
                    if (isMobile()) { sidebar.classList.add('translate-x-0', 'z-5'); sidebar.classList.remove('-translate-x-full'); }
                }

                function closeSidebar() {
                    isSidebarOpen = false;
                    hamburgerLines.forEach(h => { h.classList.add('collapseWidth'); h.classList.remove('expandWidth'); });
                    musicNote.classList.add('invisible'); musicNote.classList.remove('bouncyNote');
                    toggleSidebarBtns.forEach(btn => btn.classList.add('hidden'));
                    notToggleSidebarBtns.forEach(btn => btn.classList.remove('hidden'));
                    sidebar.classList.remove('pt-5', 'px-3');
                    if (isMobile()) { sidebar.classList.add('-translate-x-full'); sidebar.classList.remove('translate-x-0', 'z-5'); }
                }

                function isMobile() { return window.innerWidth < MOBILE_WIDTH; }

                function handleResponsive() {
                    if (isMobile()) {
                        sidebar.classList.add('absolute');
                        if (sidebar.classList.contains('relative')) sidebar.classList.remove('relative');
                        if (!isSidebarOpen) { sidebar.classList.add('-translate-x-full'); sidebar.classList.remove('translate-x-0', 'z-5'); }
                    } else {
                        sidebar.classList.remove('absolute', '-translate-x-full'); sidebar.classList.add('translate-x-0');
                    }
                }

                hamburgerBtn.addEventListener('click', () => { isSidebarOpen ? closeSidebar() : openSidebar(); });
                window.addEventListener('resize', handleResponsive);
            }
        }

        // ==========================================
        // POPUP BEHAVIOR (EVENT DELEGATION) - FIXED
        // ==========================================
        // We MUST use delegation for Popups because changing Mood destroys the navbar buttons.

        if (window.navClickHandler) {
            document.removeEventListener('click', window.navClickHandler);
        }

        window.navClickHandler = (e) => {
            // 1. Profile Popup Toggle
            const profileBtn = e.target.closest('#profileNavbar');
            if (profileBtn) {
                togglePopup('profilePopup');
                closePopup('changeMood');
                return;
            }

            // 2. Mood Popup Toggle
            const moodBtn = e.target.closest('#currentMood');
            if (moodBtn) {
                togglePopup('changeMood');
                closePopup('profilePopup');
                return;
            }

            // 3. Close Profile Popup Button
            if (e.target.closest('#closeProfilePopup')) {
                closePopup('profilePopup');
                return;
            }

            // 4. Click Outside - Profile
            const profilePopup = getElement('profilePopup');
            if (profilePopup && !profilePopup.classList.contains('invisible')) {
                const content = profilePopup.querySelector('.popupContent');
                if (content && !content.contains(e.target) && !e.target.closest('#profileNavbar')) {
                    closePopup('profilePopup');
                }
            }

            // 5. Click Outside - Mood
            const moodPopup = getElement('changeMood');
            if (moodPopup && !moodPopup.classList.contains('invisible')) {
                const content = moodPopup.querySelector('.popupContent');
                if (content && !content.contains(e.target) && !e.target.closest('#currentMood')) {
                    closePopup('changeMood');
                }
            }

            // NOTE: Search bar click-outside is handled by the specific search listener above
        };

        document.addEventListener('click', window.navClickHandler);
    };

    // Initialize on load
    document.addEventListener('DOMContentLoaded', initNavigation);
    // Initialize on Livewire navigation
    document.addEventListener('livewire:navigated', initNavigation);
}