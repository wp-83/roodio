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

// Hamburger and Sidebar functionality
const hamburgerBtn = document.getElementById('hamburgerBtn');
const hamburgerLine = hamburgerBtn.querySelectorAll('#hamburgerLine');
const musicNote = hamburgerBtn.querySelector('#musicNote');
const sidebar = document.querySelector('#sidebar');
const notToggleSidebarBtn = sidebar.querySelectorAll('#notToggleSidebar');
const toggleSidebarBtn = sidebar.querySelectorAll('#toggleSidebar');

// console.log(toggleSidebarBtn);

// set music note invisible
document.addEventListener('DOMContentLoaded', (e) => {
    e.preventDefault();
    musicNote.classList.add('invisible');
})

// hamburger button interaction
hamburgerBtn.addEventListener('click', (e) => {
    //prevent default browser behaviour
    e.preventDefault();

    // hamburger animation
    hamburgerLine.forEach(h => {
        if (h.classList.contains('expandWidth')) {
            h.classList.add('collapseWidth');
            h.classList.remove('expandWidth');
        } else {
            h.classList.add('expandWidth');
            h.classList.remove('collapseWidth');
        }
    });

    // visibility animation of music note
    if (musicNote.classList.contains('invisible')){
        musicNote.classList.remove('invisible');
    } else {
        musicNote.classList.add('invisible');
    }
    
    // music note animation
    musicNote.classList.toggle('bouncyNote');

    // open sidebar when hamburger expand
    toggleSidebarBtn.forEach(btn => {
        btn.classList.toggle('hidden');
    });

    // close sidebar shortcut icon when hamburger shrink
    notToggleSidebarBtn.forEach(btn => {
        btn.classList.toggle('hidden');
    });

    // give special style for sidebar container
    if (!toggleSidebarBtn[0].classList.contains('hidden')){
        sidebar.classList.add('pt-5', 'px-3');
    } else {
        sidebar.classList.remove('pt-5', 'px-3');
    }

    // sidebar when < 768px view
    if(window.innerWidth < 768){
        sidebar.classList.toggle('translate-x-0', 'z-5');
    }
});

window.addEventListener('resize', (e) => {
    e.preventDefault();

    if(window.innerWidth < 768){
        sidebar.classList.add('absolute');
    } else {
        sidebar.classList.remove('absolute');
    }
});

        // <div class='flex flex-col items-center justify-center gap-2 py-1 w-max h-max relative cursor-pointer z-10 ' x-data="{ active: false }" x-on:click="active = !active">
        //     @for($i = 0; $i < 3; $i++)
        //         <div {{ $attributes->merge(["class" =>  $elementColor[$mood] . ' w-8 h-1 rounded-md hamburger-line ']) }} :class="active ? 'expandWidth' : 'collapseWidth'"></div>
        //     @endfor
        //     <div {{ $attributes->merge(["class" => 'absolute w-7 h-7 ']) }} :class="active ? 'bouncyNote' : 'invisible'">
        //         <img src="{{ asset('assets/icons/music-notes.svg') }}" alt="music-notes">
        //     </div>
        // </div>



        // sidebar
        // <div
// {{ 
//     $attributes->merge([
//         'class' => 'flex flex-col gap-5 w-fit bg-primary-85 h-full pt-2 ' . (($isToggle) ? ' px-4 ' : ' ')
//     ])
// }}