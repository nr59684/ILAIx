// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
  /* ──────────────────────────────────────────────────────────── 1 ▌Bubbles */
  const container   = document.getElementById('circleContainer');
  const numCircles  = 100;

  for (let i = 0; i < numCircles; i++) {
    const circle   = document.createElement('div');
    circle.className = 'circle';

    const size      = Math.random() * 8 + 2 + 'px';
    const startY    = Math.random() * 100 + 100 + 'vh';
    const duration  = (28 + Math.random() * 9) + 's';
    const delay     = Math.random() * 37 + 's';

    circle.style.width             = size;
    circle.style.height            = size;
    circle.style.animationDuration = duration;
    circle.style.animationDelay    = delay;
    circle.style.animationName     = `move-${i}`;

    const style = document.createElement('style');
    style.textContent = `
      @keyframes move-${i} {
        from { transform: translate(${Math.random() * 100}vw, ${startY}); }
        to   { transform: translate(${Math.random() * 100}vw, ${-startY}); }
      }
    `;
    document.head.appendChild(style);
    container.appendChild(circle);
  }

  /* ───────────────────────────────────────────────────────── 2 ▌Helpers */
  async function postData (url = '', data = {}) {
    const res = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(data)
    });
    return res.json();
  }

  /* ─────────────────────────────────────────────────────── 3 ▌Grab nodes */
  const primaryPane    = document.getElementById('primaryPane');
  const secondaryPane  = document.getElementById('secondaryPane');

  const sendButton     = document.getElementById('sendButton');
  const questionInput  = document.getElementById('questionInput');
  const question2      = document.getElementById('question2');
  const solution       = document.getElementById('solution');

  const dropdownButton = document.getElementById('dropdownButton');
  const dropdownMenu   = document.getElementById('dropdownMenu');
  const dropdownItems  = dropdownMenu.querySelectorAll('li');

  const explainButton  = document.getElementById('explainButton');
  let   explanations   = null;
  let modelReady = false;  // flag to track if model is loaded


  questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();                      // stop form submit / refresh
    if (!modelReady) {
      alert('Please select a model first.');
      return;
    }
    sendButton.click();                      // trigger normal flow
  }
  });

  /* ──────────────────────────────────────────────── 4 ▌Send main prompt */
  sendButton.addEventListener('click', async () => {

    // CLICK handler (first line)
  if (!modelReady) { alert('Please select a model first.'); return; }
  
  const raw = questionInput.value.trim();
  if (!raw) return;                    // nothing to send

  // 1. extract the license name (text before the first colon)
  const [licenseName /*unusedText*/] = raw.split(':');
  const modelName = dropdownButton.innerText.trim();

  // clear the field for a nicer UX
  questionInput.value = '';
  dropdownMenu.classList.add('hidden');   // just in case it’s still open

  // 2. ask the backend to classify & store the chat
  await postData('/api', { question: raw, modelName });

  // 3. jump to the dedicated chat page
  window.location.href = `/chat/${encodeURIComponent(licenseName.trim())}`;
});

  /* ───────────────────────────────────────────── 5 ▌Model-dropdown logic */
  dropdownButton.addEventListener('click', () =>
    dropdownMenu.classList.toggle('hidden')
  );

  dropdownItems.forEach(item => {
    item.addEventListener('click', async () => {
      const modelName = item.querySelector('strong').innerText;
      dropdownButton.innerHTML = '<strong>Loading…</strong>';
      modelReady = true;
      sendButton.disabled = false;
      sendButton.classList.remove('opacity-40','cursor-not-allowed');

      const res = await postData('/loadModel', { modelName });
      dropdownMenu.classList.add('hidden');

      if (res)   dropdownButton.innerHTML = `<strong>${modelName}</strong>`;
      else {
        alert('Error loading model');
        location.reload();
      }
    });
  });

  // Close dropdown when clicking outside
  document.addEventListener('click', e => {
    if (!dropdownButton.contains(e.target) && !dropdownMenu.contains(e.target)) {
      dropdownMenu.classList.add('hidden');
    }
  });

  /* ───────────────────────────────────────────── 6 ▌Explain button flow */
  explainButton.addEventListener('click', async () => {
    explainButton.classList.add('hidden');           // disable + hide
    const labels      = solution.innerHTML.split('<br>');
    let   newHTML     = '';

    labels.forEach((lab, idx) => {
      newHTML += `<button class="label-button" data-idx="${idx}">${lab}</button><br>`;
    });
    solution.innerHTML = newHTML;

    const licenseName = window.location.pathname.split('/').pop();
    const res = await postData('/getExplanations', {
      question: question2.textContent, licenseName
    });
    explanations = res;

    // add listeners to each label button
    document.querySelectorAll('.label-button').forEach(btn =>
      btn.addEventListener('click', () => toggleHighlight(btn.dataset.idx, btn))
    );
  });

  /* ──────────────────────────────────────────────── 7 ▌Highlight helpers */
  const usedColors = new Set();
  const getRandomLightColor = () => {
    let color;
    do {
      const hue = Math.floor(Math.random() * 360);
      color = `hsl(${hue},70%,50%)`;
    } while (usedColors.has(color));
    usedColors.add(color);
    return color;
  };

  document.getElementById('question2').addEventListener('click', e => {
    if (!e.target.classList.contains('highlighted')) return;
    const idx = e.target.dataset.idx;
    const btn = document.querySelector(`.label-button[data-idx="${idx}"]`);
    if (btn) {
      btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
      btn.classList.add('ring-2', 'ring-indigo-400');
      setTimeout(() => btn.classList.remove('ring-2', 'ring-indigo-400'), 1500);
    }
  });

  function toggleHighlight (idx, btn) {
    const licenceEl = document.getElementById('question2');
    let   html      = licenceEl.innerHTML;

    const label     = Object.keys(explanations)[idx];
    const phrases   = explanations[label];

    // assign / reuse colour
    let color = btn.dataset.color;
    if (!color) {
      color = getRandomLightColor();
      btn.dataset.color = color;
    }

    const active = btn.classList.toggle('active');
    btn.style.backgroundColor = active ? color : '#4b5563'; // slate-600 fallback
    btn.style.color           = 'white';

    phrases.forEach(txt => {
      const esc = txt.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
      const re  = new RegExp(`(${esc})`, 'gi');
      html = html.replace(
        re,
        active
          ? `<span class="highlighted" data-idx="${idx}" style="color:${color};font-weight:bold;">$1</span>`
          : `$1`
      );
    });

    licenceEl.innerHTML = html;

    if (active) {
      setTimeout(() => {
        const first = document.querySelector('#question2 .highlighted');
        first && first.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 100);
    }
  }
});
