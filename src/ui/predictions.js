// Prediction cards for fixed held-out images: the shared model's current guess, its latest
// change, and its top class probabilities. Every value comes from the real model.

const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
// The model's guess, or -1 ("?") while its top class is barely above chance, e.g. the
// untrained model, whose top class is just whichever its starting weights happen to favor.
export function guessOf(probabilities) {
  const top = Math.max(...probabilities);
  return top < 1.5 / probabilities.length ? -1 : probabilities.indexOf(top);
}

// A card's static parts: actual class, image, and slots for the live prediction.
export function createCard({ label, url }, classNames) {
  const card = element('div', 'prediction-sample');
  const actual = element('span', 'prediction-actual', `Actual: ${classNames[label]}`);
  const art = element('div', 'item-art');
  const image = element('img');
  image.src = url;
  image.alt = `Held-out test image: ${classNames[label]}`;
  art.append(image);
  const current = element('div', 'prediction-current');
  current.append(element('strong', 'prediction-name', '—'), element('span', 'prediction-result'));
  const transition = element('div', 'prediction-transition');
  const comparison = element('dl', 'prediction-comparison');
  comparison.hidden = true;
  for (const [term, className] of [['Initial', 'first-guess'], ['Final', 'final-guess']]) {
    const item = element('div');
    item.append(element('dt', '', term), element('dd', className));
    comparison.append(item);
  }
  card.append(actual, art, current, transition, comparison, element('dl', 'class-scores'));
  card.dataset.label = label;
  return card;
}

// probabilities: the latest class probabilities for this image. guesses: the model's guess
// each time it changed, starting with the untrained model's (see guessChanges).
export function updateCard(card, probabilities, guesses, classNames, { done = false } = {}) {
  const label = Number(card.dataset.label);
  const guess = guessOf(probabilities);
  const correct = guess === label;
  const name = index => index < 0 ? '?' : classNames[index];
  const tone = index => index < 0 ? 'is-unsure' : index === label ? 'is-correct' : 'is-wrong';
  card.classList.toggle('wrong', guess >= 0 && !correct);
  card.classList.toggle('unsure', guess < 0);
  card.querySelector('.prediction-name').textContent = name(guess);
  card.querySelector('.prediction-result').textContent = guess < 0 ? 'Not sure yet' : correct ? '✓ Correct' : '× Incorrect';
  // The latest time the guess changed, e.g. "Shirt → Trouser".
  const transition = card.querySelector('.prediction-transition');
  transition.replaceChildren();
  if (guesses.length > 1) {
    const [from, to] = guesses.slice(-2);
    transition.append(element('span', tone(from), name(from)), element('span', 'transition-arrow', ' → '), element('span', tone(to), name(to)));
  } else {
    transition.textContent = 'No change yet';
  }
  transition.hidden = done;
  card.querySelector('.prediction-current').hidden = done;
  const comparison = card.querySelector('.prediction-comparison');
  comparison.hidden = !done;
  const first = card.querySelector('.first-guess');
  first.textContent = name(guesses[0]);
  first.className = `first-guess ${tone(guesses[0])}`;
  const final = card.querySelector('.final-guess');
  final.textContent = name(guess);
  final.className = `final-guess ${tone(guess)}`;
  const scores = card.querySelector('.class-scores');
  if (scores) {
    // The four likeliest classes, always including the actual one.
    const ranked = probabilities.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value);
    const top = ranked.slice(0, 4);
    if (!top.some(item => item.index === label)) top[3] = ranked.find(item => item.index === label);
    top.sort((a, b) => b.value - a.value);
    // Same classes in the same order: update the rows in place so the bars slide.
    const rows = [...scores.children];
    if (rows.map(row => Number(row.dataset.class)).join() !== top.map(item => item.index).join()) {
      scores.replaceChildren(...top.map(({ index }) => scoreRow(index, classNames[index])));
    }
    top.forEach(({ value, index }, i) => {
      const row = scores.children[i];
      row.className = `class-score${index === label ? ' actual-class' : ''}${index === guess ? ' top-class' : ''}`;
      row.querySelector('i').style.width = `${value * 100}%`;
      row.querySelector('.score-value').textContent = `${Math.round(value * 100)}%`;
    });
  }
  return correct;
}

function scoreRow(index, name) {
  const row = element('div', 'class-score');
  row.dataset.class = index;
  const bar = element('span', 'score-track');
  bar.append(element('i'));
  const amount = element('dd');
  amount.append(bar, element('span', 'score-value'));
  row.append(element('dt', '', name), amount);
  return row;
}

// "1/4 → 3/4 correct on these test images", for the end of a run.
export function resultSummary(initialCorrect, finalCorrect, total) {
  const summary = document.createDocumentFragment();
  summary.append(
    element('strong', `result-before ${initialCorrect === total ? 'is-correct' : 'is-wrong'}`, `${initialCorrect}/${total}`),
    element('span', 'result-arrow', ' → '),
    element('strong', `result-after ${finalCorrect === total ? 'is-correct' : 'is-wrong'}`, `${finalCorrect}/${total}`),
    element('span', 'result-scope', 'correct on these test images'));
  return summary;
}

// One image's guesses over evaluations, kept only where they changed. Phones get just the
// first and last two, which is all a card shows.
export function guessChanges(history, image) {
  const guesses = history.map(evaluation => guessOf(evaluation[image]));
  const changes = guesses.filter((value, i) => i === 0 || value !== guesses[i - 1]);
  return changes.length > 3 ? [changes[0], ...changes.slice(-2)] : changes;
}

