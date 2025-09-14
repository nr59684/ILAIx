/* Annotate page – add / remove explanations and save corrected labels */
document.addEventListener('DOMContentLoaded', () => {

  /* ─ helpers ─ */
  const used = new Set();
  const rand = () => { let c; do { c=`hsl(${Math.random()*360|0},70%,50%)`; }
                      while (used.has(c)); used.add(c); return c; };
  const esc  = s => s.replace(/[-/\\^$*+?.()|[\]{}]/g,'\\$&');
  const rmPhrase = (lab, txt) => {
    explanations[lab] = (explanations[lab]||[]).filter(p=>p!==txt);
    if (explanations[lab].length===0) delete explanations[lab];
  };

  /* ─ elements & initial data ─ */
  const pane     = document.getElementById('textPane');
  const rows     = document.querySelectorAll('#labelList label');
  const saveBtn  = document.getElementById('saveBtn');

  const { license_name, explanations = {} } =
        JSON.parse(document.getElementById('initialData').textContent);

  const colour  = {};               // label → colour
  const wrapped = new Set();        // labels already wrapped
  let   active  = null;             // current active label

  /* ─ palette row behaviour ─ */
  rows.forEach(row => {
    const lab = row.dataset.label;
    const box = row.querySelector('input[type=checkbox]');
    const dot = row.querySelector('span.w-3');
    const txt = row.querySelector('span.text-sm');
    colour[lab] = colour[lab] || rand();

    row.addEventListener('click', e => {
      if (e.target === box) return;          // let checkbox work
      e.preventDefault();                    // stop label→checkbox toggle

      active = lab;
      row.classList.toggle('active');
      const on  = row.classList.contains('active');
      const clr = colour[lab];

      if (on) {
        row.style.backgroundColor = clr + '33';
        dot && (dot.style.backgroundColor = clr);
        txt && (txt.style.color = clr);
      } else {
        row.style.backgroundColor = '';
        dot && (dot.style.backgroundColor = 'rgb(107 114 128)');
        txt && (txt.style.color = '');
      }

      if (on && !wrapped.has(lab) && explanations[lab]) {
        explanations[lab].forEach(ph => wrapFirstMatch(ph, lab));
        wrapped.add(lab);
      }

      /* colour-on / colour-off spans */
      pane.querySelectorAll(`span.hl[data-lab="${lab}"]`)
          .forEach(s=>{
            s.style.color      = on ? clr : '';
            s.style.fontWeight = on ? '600' : '';
          });

      if (on) {
        const first = pane.querySelector(`span.hl[data-lab="${lab}"]`);
        first && first.scrollIntoView({behavior:'smooth',block:'center'});
      }
    });
  });

  /* ─ add new phrase with selection ─ */
  pane.addEventListener('mouseup', () => {
    if (!active) return;
    const sel   = window.getSelection();
    const range = sel.rangeCount ? sel.getRangeAt(0) : null;
    const text  = sel.toString().trim();
    if (!range || !text) return;
    sel.removeAllRanges();

    explanations[active] = explanations[active] || [];
    if (explanations[active].includes(text)) return;
    explanations[active].push(text);

    const span = document.createElement('span');
    span.className   = 'hl';
    span.dataset.lab = active;
    span.style.cssText = `color:${colour[active]};font-weight:600;`;
    try { range.surroundContents(span); }
    catch { span.appendChild(range.extractContents()); range.insertNode(span); }

    const row = document.querySelector(`#labelList label[data-label="${active}"]`);
    const dot = row && row.querySelector('span.w-3');
    row && row.classList.add('active');
    dot && (dot.style.backgroundColor = colour[active]);
  });

  /* ─ shift-click span = delete ─ */
  pane.addEventListener('click', e => {
    if (!e.target.classList.contains('hl')) return;
    const lab = e.target.dataset.lab;
    if (e.shiftKey) {
      const phrase = e.target.textContent;
      rmPhrase(lab, phrase);
      e.target.replaceWith(document.createTextNode(phrase));

      if (!explanations[lab]) {
        const row = document.querySelector(`#labelList label[data-label="${lab}"]`);
        const dot = row && row.querySelector('span.w-3');
        const txt = row && row.querySelector('span.text-sm');
        row   && row.classList.remove('active');
        row   && (row.style.backgroundColor='');
        dot   && (dot.style.backgroundColor='rgb(107 114 128)');
        txt   && (txt.style.color='');
      }
      return;
    }

    /* normal click → scroll palette row */
    const row = document.querySelector(`#labelList label[data-label="${lab}"]`);
    row && row.scrollIntoView({behavior:'smooth',block:'center'});
    row && row.classList.add('ring-2','ring-indigo-400');
    setTimeout(()=> row.classList.remove('ring-2','ring-indigo-400'),1200);
  });

  /* ─ save to backend ─ */
  saveBtn.addEventListener('click', async () => {
    saveBtn.disabled=true; saveBtn.textContent='Saving…';

    /* get labels ticked by reviewer */
    const selected = Array.from(
          document.querySelectorAll('#labelList input[type=checkbox]:checked')
    ).map(cb => cb.closest('label').dataset.label);

    /* keep explanations only for selected labels */
    const filtered = {};
    selected.forEach(l => { if (explanations[l]) filtered[l] = explanations[l]; });

    const res = await fetch(`/annotate/${encodeURIComponent(license_name)}`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({explanations:filtered, labels:selected})
    });

    if (res.ok){
      saveBtn.textContent='Saved ✔';
      setTimeout(()=>{ saveBtn.textContent='Save'; saveBtn.disabled=false;},1500);
      /* turn status dot green */
      selected.forEach(lab=>{
        const dot = document.querySelector(`#labelList label[data-label="${lab}"] span.w-3`);
        dot && (dot.style.backgroundColor='#22c55e');
      });
    } else {
      alert('Error saving');
      saveBtn.disabled=false; saveBtn.textContent='Save';
    }
  });

  /* helper: wrap first occurrence of phrase ----------------------------- */
  function wrapFirstMatch(ph, lab){
    const rx=new RegExp(ph.replace(/[-/\\^$*+?.()|[\]{}]/g,'\\$&'),'i');
    const walk=document.createTreeWalker(pane,NodeFilter.SHOW_TEXT,null);
    while(walk.nextNode()){
      const node=walk.currentNode, ix=node.nodeValue.search(rx);
      if(ix>=0){
        const rng=document.createRange();
        rng.setStart(node,ix); rng.setEnd(node,ix+ph.length);
        const sp=document.createElement('span');
        sp.className='hl'; sp.dataset.lab=lab;
        sp.style.cssText=`color:${colour[lab]};font-weight:600;`;
        rng.surroundContents(sp); return;
      }
    }
  }
});
