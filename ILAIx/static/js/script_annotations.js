/* ============================================================================
   Annotation table helpers
   ---------------------------------------------------------------------------
   – Live-filters the rows in annotations.html via:
     • Search box (license name or model)
     • “Show only incomplete” toggle
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {


  const searchInput      = document.getElementById('searchInput');
  const toggleIncomplete = document.getElementById('toggleIncomplete');
  const rows             = Array.from(document.querySelectorAll('#jobsTable tr'));

  function applyFilters () {
    const term           = (searchInput.value || '').toLowerCase().trim();
    const onlyIncomplete = toggleIncomplete.checked;

    rows.forEach(tr => {
      const matchesSearch =
        tr.dataset.license.includes(term) || tr.dataset.model.includes(term);

      const matchesIncomplete =
        !onlyIncomplete || tr.dataset.done === '0'; // show only rows without ✔

      tr.style.display = (matchesSearch && matchesIncomplete) ? '' : 'none';
    });
  }

  /* ── event wiring ─────────────────────────────────────────────────────── */
  searchInput.addEventListener('input',  applyFilters);
  toggleIncomplete.addEventListener('change', applyFilters);

  /* ── initial state ────────────────────────────────────────────────────── */
  applyFilters();

  const container = document.getElementById('circleContainer');
    const numCircles = 100;
  
    for (let i = 0; i < numCircles; i++) {
      const circle = document.createElement('div');
      circle.className = 'circle';
      
      // Random styling
      const size = Math.random() * 8 + 2 + 'px';
      const startY = Math.random() * 100 + 100 + 'vh';
      const duration = (28 + Math.random() * 9) + 's';
      const delay = Math.random() * 37 + 's';
      
      circle.style.width = size;
      circle.style.height = size;
      circle.style.animationDuration = duration;
      circle.style.animationDelay = delay;
      circle.style.animationName = `move-${i}`;
      
      // Create unique keyframe animation
      const style = document.createElement('style');
      style.textContent = `
        @keyframes move-${i} {
          from { transform: translate(${Math.random() * 100}vw, ${startY}); }
          to { transform: translate(${Math.random() * 100}vw, ${-startY}); }
        }
      `;
      document.head.appendChild(style);
      
      container.appendChild(circle);
    }
});


