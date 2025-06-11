// Initialize hover swap functionality
document.addEventListener('DOMContentLoaded', function() {
  // Hover swap for icons
  document.querySelectorAll('.hover-swap').forEach(img => {
    img.addEventListener('mouseenter', () => {
      const hoverSrc = img.getAttribute('data-hover');
      if (hoverSrc) {
        img.src = hoverSrc;
      }
    });

    img.addEventListener('mouseleave', () => {
      const originalSrc = img.getAttribute('data-src');
      if (originalSrc) {
        img.src = originalSrc;
      }
    });
  });

  // Custom scrollbar functionality
  const scrollable = document.querySelector('.left-scrollable');
  const thumb = document.querySelector('.scroll-thumb');
  const track = document.querySelector('.scroll-divider');

  if (scrollable && thumb && track) {
    function updateThumbPosition() {
      const scrollRatio = scrollable.scrollTop / (scrollable.scrollHeight - scrollable.clientHeight);
      const thumbHeight = Math.max(scrollable.clientHeight / scrollable.scrollHeight * track.clientHeight, 30);
      const thumbTop = scrollRatio * (track.clientHeight - thumbHeight);

      thumb.style.height = `${thumbHeight}px`;
      thumb.style.top = `${thumbTop}px`;
    }

    scrollable.addEventListener('scroll', updateThumbPosition);
    window.addEventListener('resize', updateThumbPosition);
    window.addEventListener('load', updateThumbPosition);

    // Enable drag on thumb
    let isDragging = false;
    let startY, startTop;

    thumb.addEventListener('mousedown', (e) => {
      isDragging = true;
      startY = e.clientY;
      startTop = parseInt(thumb.style.top) || 0;
      thumb.style.cursor = 'grabbing';
      document.body.style.userSelect = 'none';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const delta = e.clientY - startY;
      const newTop = Math.min(track.clientHeight - thumb.offsetHeight, Math.max(0, startTop + delta));
      const scrollRatio = newTop / (track.clientHeight - thumb.offsetHeight);
      scrollable.scrollTop = scrollRatio * (scrollable.scrollHeight - scrollable.clientHeight);
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
      thumb.style.cursor = 'grab';
      document.body.style.userSelect = '';
    });
  }
});

// Global functions for dropdown functionality
function toggleDropdown(button) {
  const dropdown = button.parentElement;
  dropdown.classList.toggle('open');
  
  // Close other dropdowns
  document.querySelectorAll('.dropdown').forEach(otherDropdown => {
    if (otherDropdown !== dropdown && otherDropdown.classList.contains('open')) {
      otherDropdown.classList.remove('open');
    }
  });
}

// Select all functionality for dropdowns
function toggleAllCheckboxes(groupName) {
  const selectAllCheckbox = document.getElementById(`select-all-${groupName}`);
  const checkboxes = document.querySelectorAll(`input[name="${groupName}"]`);

  checkboxes.forEach((checkbox) => {
    checkbox.checked = selectAllCheckbox.checked;
  });
}

// Close dropdowns when clicking outside
document.addEventListener('click', function(event) {
  if (!event.target.closest('.dropdown')) {
    document.querySelectorAll('.dropdown').forEach(dropdown => {
      dropdown.classList.remove('open');
    });
  }
});