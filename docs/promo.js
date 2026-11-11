// Quiet inline preview; a deliberate click opens the full film.
(() => {
  'use strict';
  const preview = document.getElementById('promo-preview');
  const player = document.getElementById('promo-player');
  const shell = document.getElementById('promo-shell');
  const open = document.getElementById('promo-open');
  const dialog = document.getElementById('promo-dialog');
  const close = document.getElementById('promo-close');
  const error = document.getElementById('promo-error');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let inView = false;
  let firstWatch = true;
  let previousOverflow = '';
  preview.muted = true;
  const stopped = () => reduced.matches || document.body.classList.contains('motion-paused');
  function syncPreview() {
    const allowed = inView && !document.hidden && !dialog.open && !stopped();
    if (!allowed) { preview.pause(); return; }
    if (!preview.getAttribute('src')) preview.src = 'media/promo/llm4robot-bench-preview.mp4?v=b73f1d1d94a6';
    preview.play().catch(() => shell.classList.add('promo-idle'));
  }
  preview.addEventListener('playing', () => shell.classList.remove('promo-idle'));
  preview.addEventListener('error', () => shell.classList.add('promo-idle'));
  new IntersectionObserver(entries => {
    inView = entries[0].isIntersecting;
    syncPreview();
  }, { threshold: .12 }).observe(shell);
  new MutationObserver(syncPreview).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) player.pause();
    syncPreview();
  });
  reduced.addEventListener('change', syncPreview);
  open.addEventListener('click', () => {
    preview.pause();
    previousOverflow = document.body.style.overflow;
    dialog.showModal();
    document.body.style.overflow = 'hidden';
    error.hidden = true;
    // First deliberate watch begins at the opening, with music enabled.
    if (firstWatch) { player.currentTime = 0; player.muted = false; firstWatch = false; }
    player.play().catch(() => { /* Native controls remain available. */ });
  });
  const closeFilm = () => { player.pause(); dialog.close(); };
  close.addEventListener('click', closeFilm);
  dialog.addEventListener('cancel', () => player.pause());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) closeFilm();
  });
  dialog.addEventListener('close', () => {
    player.pause();
    document.body.style.overflow = previousOverflow;
    open.focus({ preventScroll: true });
    syncPreview();
  });
  player.addEventListener('error', () => { error.hidden = false; });
  player.querySelector('source').addEventListener('error', () => { error.hidden = false; });
})();
