import './style.css';
import { api } from './api.js';
import { Viewer } from './viewer.js';
import { Annotations } from './annotations.js';
import { FlagList } from './flags.js';

const els = {
  scanSelect: document.getElementById('scan-select'),
  scanMeta: document.getElementById('scan-meta'),
  modalityRow: document.getElementById('modality-row'),
  viewChips: document.querySelectorAll('.chip[data-view]'),
  clipRange: document.getElementById('clip-range'),
  checkList: document.getElementById('check-list'),
  runChecks: document.getElementById('run-checks'),
  loadGallery: document.getElementById('load-gallery'),
  runStatus: document.getElementById('run-status'),
  flagsList: document.getElementById('flags-list'),
  flagsCount: document.getElementById('flags-count'),
  detail: document.getElementById('detail-panel'),
};

const viewer = new Viewer(document.getElementById('gl'), document.getElementById('viewer-hud'));
const annotations = new Annotations(viewer.nv);

const state = { scans: [], scanMeta: {}, flags: [] };

const flagList = new FlagList({
  listEl: els.flagsList,
  countEl: els.flagsCount,
  detailEl: els.detail,
  onSelect: (f) => selectFlag(f),
  onAdjudicate: (rec) => api.adjudicate(rec).catch((e) => console.warn('adjudicate failed', e)),
});

// --- annotations for the currently loaded scan --------------------------------

async function renderAnnotationsForCurrentScan() {
  const forScan = state.flags.filter((f) => f.scan_id === viewer.currentScan);
  await annotations.renderAll(forScan);
}

// --- scan loading -------------------------------------------------------------

// Loads the base volume only. Annotation rendering is the caller's decision
// (overview via renderAnnotationsForCurrentScan, or isolate via renderOne).
async function loadScan(scanId, modality) {
  const meta = state.scanMeta[scanId] || (state.scanMeta[scanId] = await api.scan(scanId));
  const mod = modality || meta.default_modality;
  await viewer.loadScan(scanId, mod);
  els.scanSelect.value = scanId;
  buildModalityChips(meta, mod);
  renderScanMeta(meta, mod);
}

function buildModalityChips(meta, active) {
  els.modalityRow.innerHTML = '';
  for (const m of meta.modalities) {
    const b = document.createElement('button');
    b.className = 'chip' + (m === active ? ' active' : '');
    b.textContent = m;
    b.addEventListener('click', async () => {
      await viewer.loadScan(meta.scan_id, m);
      buildModalityChips(meta, m);
      renderScanMeta(meta, m);
      await renderAnnotationsForCurrentScan();
    });
    els.modalityRow.appendChild(b);
  }
}

function renderScanMeta(meta, modality) {
  const m = meta.meta?.[modality];
  if (!m) { els.scanMeta.textContent = ''; return; }
  els.scanMeta.textContent =
    `site   ${meta.site || '-'}\n` +
    `shape  ${m.shape.join(' x ')}\n` +
    `spacing ${m.zooms_mm.join(' x ')} mm\n` +
    `orient ${m.orientation}\n` +
    `dtype  ${m.dtype}`;
}

// --- flags --------------------------------------------------------------------

// Selections are serialized: overlapping async renders would otherwise leave
// orphaned overlays on the canvas.
let selectChain = Promise.resolve();

function selectFlag(f) {
  selectChain = selectChain.then(() => _selectFlag(f)).catch((e) => console.warn('select failed', e));
  return selectChain;
}

async function _selectFlag(f) {
  if (f.scan_id !== viewer.currentScan) {
    await loadScan(f.scan_id);
  }
  // Isolate this flag's annotation so a busy scan does not obscure it.
  await annotations.renderOne(f);
  viewer.flyTo(f.location?.world_mm);
}

async function setFlags(flags) {
  state.flags = flags;
  flagList.setFlags(flags);
  await renderAnnotationsForCurrentScan();
}

// --- controls -----------------------------------------------------------------

function selectedCheckIds() {
  return [...els.checkList.querySelectorAll('input:checked')].map((i) => i.value);
}

async function runChecks() {
  els.runChecks.disabled = true;
  els.runStatus.textContent = 'Running checks over the cohort...';
  try {
    const ids = selectedCheckIds();
    const { flags } = await api.run(null, ids.length ? ids : null);
    await setFlags(flags);
    els.runStatus.textContent = `${flags.length} flag(s) across the cohort.`;
    if (flags.length) await selectFlag(flags[0]);
  } catch (e) {
    els.runStatus.textContent = `Error: ${e.message}`;
  } finally {
    els.runChecks.disabled = false;
  }
}

async function loadGallery() {
  els.runStatus.textContent = 'Building annotation gallery...';
  const { flags } = await api.gallery();
  if (flags.length) await loadScan(flags[0].scan_id);
  await setFlags(flags); // gallery is a showcase: keep every payload type on screen
  els.runStatus.textContent = `Gallery: ${flags.length} annotation type(s) on ${flags[0]?.scan_id || '-'}.`;
  if (flags.length) viewer.flyTo(flags[0].location?.world_mm);
}

function wireControls() {
  els.scanSelect.addEventListener('change', async () => {
    await loadScan(els.scanSelect.value);
    await renderAnnotationsForCurrentScan(); // show all of the chosen scan's flags
  });
  els.viewChips.forEach((chip) =>
    chip.addEventListener('click', () => {
      els.viewChips.forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      viewer.setView(chip.dataset.view);
    }),
  );
  els.clipRange.addEventListener('input', () => viewer.setClip(parseFloat(els.clipRange.value)));
  els.runChecks.addEventListener('click', runChecks);
  els.loadGallery.addEventListener('click', loadGallery);
}

// --- boot ---------------------------------------------------------------------

async function boot() {
  wireControls();
  document.querySelector('.chip[data-view="multi"]').classList.add('active');

  state.scans = await api.scans();
  for (const s of state.scans) {
    const opt = document.createElement('option');
    opt.value = s.scan_id;
    opt.textContent = `${s.scan_id}  (${s.site || s.source})`;
    els.scanSelect.appendChild(opt);
  }

  const checks = await api.checks();
  els.checkList.innerHTML = checks.length
    ? ''
    : '<div class="empty-hint">No checks registered yet (Phase 1).</div>';
  for (const c of checks) {
    const label = document.createElement('label');
    label.className = 'check-item';
    label.innerHTML = `
      <input type="checkbox" value="${c.check_id}" checked />
      <span><span class="ci-id">${c.check_id}</span><br><span class="ci-desc">${c.description}</span></span>`;
    els.checkList.appendChild(label);
  }

  if (state.scans.length) await loadScan(state.scans[0].scan_id);
}

boot();

// Handle exposed for the screenshot/E2E harness to drive deterministic actions.
window.sfg = { viewer, annotations, flagList, state, loadScan, loadGallery, runChecks, setFlags, api };
