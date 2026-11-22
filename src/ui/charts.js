// SVG charts for the big screen and phone, drawn from real training history.
// Layout and styling follow the approved design (mockup.html).

const escape = text => String(text).replace(/[&<>"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[char]);
const clock = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}.${Math.floor(seconds * 10) % 10}`;

// A round number of seconds for the x axis, with at most ~6 ticks.
function timeAxis(seconds) {
  const step = [5, 10, 20, 30, 60, 120].find(size => seconds / size <= 6) ?? 300;
  const max = Math.max(step * 2, Math.ceil(seconds / step) * step);
  return { max, ticks: Array.from({ length: max / step + 1 }, (_, i) => i * step) };
}

// Log-scale range for the loss, padded and snapped to readable ticks.
function lossAxis(losses) {
  const candidates = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
  let low = Math.min(...losses) * 0.85, high = Math.max(...losses) * 1.15;
  let ticks = candidates.filter(value => value >= low && value <= high);
  if (ticks.length < 2) {
    low = Math.min(low, candidates.findLast(value => value <= low) ?? low);
    high = Math.max(high, candidates.find(value => value >= high) ?? high);
    ticks = candidates.filter(value => value >= low && value <= high);
  }
  return { low, high, ticks };
}

// points: [{ seconds, loss, accuracy (held-out, 0–1), shown (correct count or undefined) }].
// Returns the chart's height, so the probability history can match it.
export function drawLossChart(svg, points, shownTotal) {
  const width = svg.clientWidth || 1000;
  const height = Math.max(260, Math.min(340, Math.round(width * 0.48)));
  const left = 58, right = 62, top = 36, bottom = height - 44;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  if (!points.length) { svg.innerHTML = ''; return height; }
  const now = points.at(-1).seconds;
  const time = timeAxis(now);
  const loss = lossAxis(points.map(point => point.loss));
  const x = seconds => left + seconds / time.max * (width - left - right);
  const y = value => bottom - Math.log(value / loss.low) / Math.log(loss.high / loss.low) * (bottom - top);
  const yAccuracy = value => bottom - value * (bottom - top);

  let grid = '<g class="plot-axis">';
  for (const seconds of time.ticks) {
    grid += `<path d="M${x(seconds)},${top}V${bottom}" stroke="#2c363e" stroke-width=".6"/><path d="M${x(seconds)},${bottom}v5" stroke="#82919c" stroke-width=".8"/><text x="${x(seconds)}" y="${bottom + 19}" text-anchor="middle">${seconds}</text>`;
  }
  for (const value of loss.ticks) {
    grid += `<path d="M${left},${y(value)}H${width - right}" stroke="#35414a" stroke-width=".6" stroke-dasharray="2 4"/><path d="M${left - 5},${y(value)}h5" stroke="#82919c" stroke-width=".8"/><text x="${left - 10}" y="${y(value) + 4}" text-anchor="end">${value}</text>`;
  }
  grid += `<path d="M${left},${top}H${width - right}V${bottom}" fill="none" stroke="#35414a" stroke-width=".7"/><path d="M${left},${top}V${bottom}H${width - right}" fill="none" stroke="#82919c" stroke-width=".8"/>
    <text class="axis-title" transform="translate(13 ${(top + bottom) / 2}) rotate(-90)" text-anchor="middle">Held-out loss (log scale)</text>
    <text class="axis-title" x="${(left + width - right) / 2}" y="${height - 5}" text-anchor="middle">Elapsed time (s)</text>`;
  const hasAccuracy = points.some(point => point.accuracy !== undefined);
  if (hasAccuracy) {
    grid += `<g class="accuracy-axis">${[0, 0.25, 0.5, 0.75, 1].map(value => `<path d="M${width - right},${yAccuracy(value)}h5" stroke="var(--accuracy)" stroke-width=".8"/><text x="${width - right + 9}" y="${yAccuracy(value) + 4}">${value * 100}%</text>`).join('')}
      <text class="axis-title" transform="translate(${width - 12} ${(top + bottom) / 2}) rotate(90)" text-anchor="middle">Accuracy</text></g>`;
  }
  grid += '</g>';

  const line = (values, at) => values.map((point, i) => `${i ? 'L' : 'M'}${x(point.seconds)},${at(point)}`).join(' ');
  const latest = points.at(-1);
  const compact = width < 600;
  const fontSize = compact ? 10 : 12;
  let legend = `<path d="M${x(0)},8h14" stroke="var(--trace)" stroke-width="2"/><text x="${x(0) + 20}" y="12" fill="var(--trace)">Held-out loss ${latest.loss.toFixed(3)}</text>`;
  let series = `<path class="loss-trace" d="${line(points, point => y(point.loss))}" fill="none" stroke="var(--trace)" stroke-width="2"/>
    ${points.map(point => `<circle class="loss-sample" cx="${x(point.seconds)}" cy="${y(point.loss)}" r="1.7" fill="var(--plot)" stroke="var(--trace)" stroke-width=".9"><title>${clock(point.seconds)} · held-out loss ${point.loss.toFixed(3)}</title></circle>`).join('')}
    <circle class="room-marker" cx="${x(latest.seconds)}" cy="${y(latest.loss)}" r="3" fill="var(--trace)"/>`;
  if (hasAccuracy) {
    const accuracyX = x(0) + (compact ? 150 : 210);
    legend += `<path d="M${accuracyX},8h14" stroke="var(--accuracy)" stroke-width="1.8"/><text x="${accuracyX + 20}" y="12" fill="var(--accuracy)">Held-out accuracy ${(latest.accuracy * 100).toFixed(1)}%</text>`;
    series += `<path class="accuracy-trace" d="${line(points, point => yAccuracy(point.accuracy))}" fill="none" stroke="var(--accuracy)" stroke-width="1.8"/>`;
  }
  // Accuracy on the few images shown above: a different, much noisier number than held-out
  // accuracy, so it gets its own dashed step line and label.
  const shown = points.filter(point => point.shown !== undefined);
  if (shown.length) {
    const path = shown.map((point, i) => `${i ? 'H' : 'M'}${x(point.seconds)}${i ? 'V' : ','}${yAccuracy(point.shown / shownTotal)}`).join(' ');
    legend += `<path d="M${x(0)},28h7v-4h7" fill="none" stroke="var(--shown)" stroke-width="1.4" stroke-dasharray="3 2"/><text class="accuracy-readout" x="${x(0) + 20}" y="30" fill="var(--shown)">Shown images ${shown.at(-1).shown}/${shownTotal}</text>`;
    series += `<path class="shown-trace" d="${path}" fill="none" stroke="var(--shown)" stroke-width="1.4" stroke-dasharray="3 2"/>`;
  }
  svg.innerHTML = `${grid}<g class="loss-legend" font-size="${fontSize}" font-family="var(--mono)">${legend}</g>
    <path d="M${x(now)},${top}V${bottom}" stroke="#82909b" stroke-width=".7" stroke-dasharray="2 4"/>${series}`;
  return height;
}

const heatColors = [[22, 26, 30], [44, 54, 61], [66, 82, 91], [96, 116, 126], [143, 161, 168]];
function heatColor(value) {
  const position = Math.min(Math.max(value, 0), 1) * (heatColors.length - 1);
  const index = Math.min(Math.floor(position), heatColors.length - 2);
  return `rgb(${heatColors[index].map((channel, i) => Math.round(channel + (heatColors[index + 1][i] - channel) * (position - index))).join(' ')})`;
}

// evaluations: [[probabilities per class] per shown image] per evaluation. Rows are classes
// grouped by image; the winning class of each evaluation is outlined green or red.
export function drawProbabilityHistory(svg, evaluations, labels, classNames, height) {
  const width = svg.clientWidth || 420;
  const left = 82, right = 14, top = 18, bottom = height - 52, gap = 5;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  if (!evaluations.length) { svg.innerHTML = ''; return; }
  // Keep cells visible on long runs by sampling at most 80 evaluations, evenly.
  const columns = evaluations.length <= 80 ? evaluations.map((_, i) => i)
    : Array.from({ length: 80 }, (_, i) => Math.round(i * (evaluations.length - 1) / 79));
  const classes = classNames.length;
  const rowHeight = (bottom - top - gap * (labels.length - 1)) / (labels.length * classes);
  const columnWidth = (width - left - right) / columns.length;
  // The actual class's row is labelled in green, which also names the image. Too dense for
  // every class name (e.g. 10 classes): label only that row. Long names are shortened.
  const dense = rowHeight < 9;
  const short = text => text.length > 11 ? `${text.slice(0, 10).trimEnd()}…` : text;
  const groups = labels.map((actual, image) => {
    const groupY = top + image * (classes * rowHeight + gap);
    const rows = classNames.map((name, classIndex) => {
      const y = groupY + classIndex * rowHeight;
      const cells = columns.map((evaluation, column) => {
        const probabilities = evaluations[evaluation][image];
        const score = probabilities[classIndex];
        const winner = probabilities.indexOf(Math.max(...probabilities)) === classIndex;
        return `<rect x="${left + column * columnWidth + 0.5}" y="${y + 0.5}" width="${Math.max(columnWidth - 1, 0.5)}" height="${Math.max(rowHeight - 1, 0.5)}" fill="${heatColor(score)}" stroke="${winner ? classIndex === actual ? 'var(--green)' : 'var(--red)' : 'none'}" stroke-width="1" stroke-opacity=".5"><title>${escape(classNames[actual])} image · eval ${evaluation} · P(${escape(name)}) = ${(score * 100).toFixed(1)}%</title></rect>`;
      }).join('');
      const label = !dense || classIndex === actual
        ? `<text x="${left - 8}" y="${y + rowHeight / 2 + 3}" text-anchor="end" fill="${classIndex === actual ? 'var(--green)' : '#9ba8b2'}">${escape(short(name))}<title>${escape(name)}</title></text>` : '';
      return label + cells;
    }).join('');
    return `<path d="M${left - 4},${groupY + 0.5}V${groupY + classes * rowHeight - 0.5}" stroke="#35414a"/>${rows}`;
  }).join('');
  const ticks = columns.map((evaluation, column) => ({ evaluation, column }))
    .filter((_, i) => i % Math.max(1, Math.ceil(columns.length / 5)) === 0)
    .map(({ evaluation, column }) => `<text x="${left + (column + 0.5) * columnWidth}" y="${bottom + 14}" text-anchor="middle">${evaluation}</text>`).join('');
  const center = (left + width - right) / 2;
  svg.innerHTML = `<defs><linearGradient id="probability-scale">${heatColors.map((_, i) => `<stop offset="${i / (heatColors.length - 1)}" stop-color="${heatColor(i / (heatColors.length - 1))}"/>`).join('')}</linearGradient></defs>
    <g font-family="var(--mono)" font-size="10" fill="#9ba8b2">
      <text x="8" y="10" font-size="9">image / class</text><text x="${width - right}" y="10" font-size="9" text-anchor="end">outline = prediction</text>
      ${groups}${ticks}
      <text x="${center}" y="${height - 24}" font-size="11" text-anchor="middle">Evaluation (round)</text>
      <rect x="${center - 45}" y="${height - 11}" width="90" height="6" fill="url(#probability-scale)"/>
      <text x="${center - 51}" y="${height - 5}" font-size="9" text-anchor="end">0%</text><text x="${center + 51}" y="${height - 5}" font-size="9">100%</text>
    </g>`;
}

// samples: [{ seconds, rate }] measured by this phone, over the last 30 seconds.
export function drawRateSparkline(svg, samples, now) {
  const window = 30;
  const recent = samples.filter(sample => sample.seconds >= now - window);
  const peak = Math.max(1, ...recent.map(sample => sample.rate));
  const magnitude = 10 ** Math.floor(Math.log10(peak));
  const max = Math.ceil(peak / magnitude) * magnitude;
  const x = seconds => 32 + (seconds - (now - window)) / window * 324;
  const y = rate => 42 - rate / max * 36;
  const path = recent.map((sample, i) => `${i ? 'L' : 'M'}${x(sample.seconds)},${y(sample.rate)}`).join(' ');
  const last = recent.at(-1);
  svg.innerHTML = `<g font-size="9" font-family="var(--mono)" fill="#9ba8b2">
      ${[0, max / 2, max].map(value => `<path d="M32 ${y(value)}H356" stroke="#35414a" stroke-width=".6" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/><path d="M28 ${y(value)}h4" stroke="#82919c" stroke-width=".7" vector-effect="non-scaling-stroke"/><text x="25" y="${y(value) + 3}" text-anchor="end">${value.toLocaleString('en-US', { notation: 'compact', maximumFractionDigits: 1 })}</text>`).join('')}
    </g><path d="M32 6V42H356" fill="none" stroke="#82919c" stroke-width=".7" vector-effect="non-scaling-stroke"/>
    <path class="rate-trace" d="${path}" fill="none" stroke="var(--orange)" stroke-width="1.2" vector-effect="non-scaling-stroke"/>
    ${last ? `<circle cx="${x(last.seconds)}" cy="${y(last.rate)}" r="1.8" fill="var(--orange)"/>` : ''}`;
}

export { clock };
