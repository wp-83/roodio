const sectionClass = ['identity-section', 'otp-section', 'account-section'];

document.addEventListener('DOMContentLoaded', () => {
  //change style for field dob, gender, and country when they have been filled
  const setupSelect = (id) => {
    const el = document.getElementById(id);
    if (!el) return;

    const update = () => {
      if (el.value) {
        el.classList.add('bg-accent-20/60', 'text-shadedOfGray-100');
        el.classList.remove('text-shadedOfGray-400', 'italic');
      } else {
        el.classList.remove('bg-accent-20/60', 'text-shadedOfGray-100');
        el.classList.add('text-shadedOfGray-400', 'italic');
      }
    };

    update();
    el.addEventListener('change', update);
  };

  setupSelect('gender');
  setupSelect('country');
  
  // modification style for date-picker
  const input = document.getElementById('default-datepicker');
  if (!input) return;

  new Datepicker(input, {
    autohide: true,
  });
});

// identity form validation
const identityFormElement = document.querySelector("#identity").querySelectorAll("input, select");
const identityBtn = document.getElementById('identityBtn');

identityBtn.addEventListener('click', () => {
  // check whether the value of each element is null
  // for (element of identityFormElement){
  //   if (element.value == ""){
  //     return false;
  //   }
  // }

  document.querySelectorAll(`.${sectionClass[0]}`).forEach(element => {
    element.classList.add('hidden');
  });

  document.querySelectorAll(`.${sectionClass[1]}`).forEach(element => {
    element.classList.remove('hidden');
  })
});
