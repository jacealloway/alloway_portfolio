const courseContainers = document.querySelectorAll('.course-container');

// Only open on click (no toggle behavior)
courseContainers.forEach(container => {
  container.addEventListener('click', function (e) {
    e.stopPropagation();

    // If you clicked *directly* on the container, open it
    // If it's already open, do nothing (no toggle!)
    if (!this.classList.contains('open')) {
      closeAllContainers();
      this.classList.add('open');
    }
  });
});

// Global click listener to close any open container
document.addEventListener('click', function (e) {
  // If click is outside any .course-container, close all
  if (!e.target.closest('.course-container')) {
    closeAllContainers();
  }
});

function closeAllContainers() {
    document.querySelectorAll('.course-container.open')
      .forEach(container => container.classList.remove('open'));
  }




  //FADING DIVS IN/OUT

const coursefaders = document.querySelectorAll('.course-container');

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    } else {
      entry.target.classList.remove('visible'); // Optional: for fade out
      entry.target.classList.remove('open');
    }
  });
}, {
  threshold: 0.1
});

coursefaders.forEach(fader => {
  observer.observe(fader);
});

coursefaders.forEach((fader, index) => {
    setTimeout(() => {
      observer.observe(fader);
    }, index * 50); // 50ms delay between each observer activation
  });



const faders = document.querySelectorAll('.fade-div');


faders.forEach(fader => {
    observer.observe(fader);
  });
  
  faders.forEach((fader, index) => {
      setTimeout(() => {
        observer.observe(fader);
      }, index * 100); // 50ms delay between each observer activation
    });
  




// window.addEventListener('scroll', () => {
//     const scrollTop = window.scrollY; // Get the current vertical scroll position
    
//     offsetX = -scrollTop * 0.01; // Horizontal movement control
//     offsetY = -scrollTop * 0.005; // Vertical movement control (optional)


//     // Apply the background position
//     document.body.style.setProperty('--before-background-position', `${offsetX}px ${offsetY}px`);

//   });

