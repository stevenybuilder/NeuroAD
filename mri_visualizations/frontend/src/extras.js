// Renders a flag's open `extra` bag into the detail panel. Checks attach
// structured context here (histograms, affine matrices, metric tables); the
// viewer contract does not care what is inside.

const PALETTE = ['#4c8dff', '#f0a13a', '#46c98b', '#ff6b5e', '#b98dff'];

export function renderExtra(container, extra) {
  if (!extra || Object.keys(extra).length === 0) return;

  if (extra.histogram) {
    container.appendChild(histogramChart(extra.histogram));
  }
  for (const h of extra.histograms || []) {
    if (h.title) {
      const t = document.createElement('div');
      t.className = 'hist-title';
      t.textContent = h.title;
      container.appendChild(t);
    }
    container.appendChild(histogramChart(h));
  }
  // Anything else structured: show as compact JSON so no context is lost.
  const rest = { ...extra };
  delete rest.histogram;
  delete rest.histograms;
  if (Object.keys(rest).length) {
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(rest, null, 2);
    container.appendChild(pre);
  }
}

// histogram = { edges:[...N+1], series:[{name, counts:[...N], color?}], xlabel? }
function histogramChart(h) {
  const W = 300;
  const H = 120;
  const canvas = document.createElement('canvas');
  canvas.className = 'hist';
  canvas.width = W * 2;
  canvas.height = H * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);

  const pad = { l: 6, r: 6, t: 8, b: 16 };
  const series = h.series || [];
  const n = series.length ? series[0].counts.length : 0;
  if (!n) return canvas;

  // Normalize each series to its own max so shapes are comparable.
  const norm = series.map((s) => {
    const max = Math.max(...s.counts, 1);
    return s.counts.map((c) => c / max);
  });

  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const x = (i) => pad.l + (i / (n - 1)) * plotW;
  const y = (v) => pad.t + (1 - v) * plotH;

  series.forEach((s, si) => {
    ctx.strokeStyle = s.color || PALETTE[si % PALETTE.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    norm[si].forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
  });

  // Legend.
  ctx.font = '9px monospace';
  series.forEach((s, si) => {
    const lx = pad.l + si * 92;
    ctx.fillStyle = s.color || PALETTE[si % PALETTE.length];
    ctx.fillRect(lx, H - 10, 8, 3);
    ctx.fillStyle = '#97a0b5';
    ctx.fillText(s.name, lx + 12, H - 7);
  });
  return canvas;
}
