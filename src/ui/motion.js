// Small animations for live numbers. Hidden tabs skip them (no frames run there).

// Counts an element's number up (or down) to `value` over 400 ms.
export function countTo(element, value, format = number => Math.round(number).toLocaleString('en-US')) {
  const from = Number(element.dataset.value ?? 0);
  element.dataset.value = value;
  cancelAnimationFrame(element.frame);
  if (document.hidden || from === value) {
    element.textContent = format(value);
    return;
  }
  const started = performance.now();
  const frame = now => {
    const t = Math.min((now - started) / 400, 1);
    element.textContent = format(from + (value - from) * (1 - (1 - t) ** 3));
    if (t < 1) element.frame = requestAnimationFrame(frame);
  };
  element.frame = requestAnimationFrame(frame);
}

// A short flash, e.g. when the phone count changes.
export function bump(element) {
  element.classList.remove('bump');
  void element.offsetWidth; // Restarts the animation.
  element.classList.add('bump');
}
