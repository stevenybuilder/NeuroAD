// Flag list + adjudication surface. Pure view: it renders severity-ranked flag
// cards and reports intent (select / adjudicate) back through callbacks. It owns
// no viewer or network state.

import { renderExtra } from './extras.js';

const SEV_ORDER = { critical: 3, error: 2, warn: 1, info: 0 };

export class FlagList {
  constructor({ listEl, countEl, detailEl, onSelect, onAdjudicate }) {
    this.listEl = listEl;
    this.countEl = countEl;
    this.detailEl = detailEl;
    this.onSelect = onSelect;
    this.onAdjudicate = onAdjudicate;
    this.flags = [];
    this.selected = null;
    this.decisions = new Map(); // flag key -> decision string
  }

  static key(f) {
    return `${f.check_id}::${f.scan_id}`;
  }

  setFlags(flags) {
    this.flags = [...flags].sort(
      (a, b) => (SEV_ORDER[b.severity] - SEV_ORDER[a.severity]),
    );
    this.selected = null;
    this.decisions.clear();
    this.detailEl.innerHTML = '';
    this._renderList();
  }

  _renderList() {
    this.countEl.textContent = String(this.flags.length);
    if (!this.flags.length) {
      this.listEl.innerHTML =
        '<div class="empty-hint">No flags. Either the cohort is clean for the selected checks, or run a check that induces a failure.</div>';
      return;
    }
    this.listEl.innerHTML = '';
    for (const f of this.flags) {
      this.listEl.appendChild(this._card(f));
    }
  }

  _card(f) {
    const key = FlagList.key(f);
    const el = document.createElement('div');
    el.className = `flag-card sev-${f.severity}`;
    if (this.selected && FlagList.key(this.selected) === key) el.classList.add('selected');
    if (this.decisions.has(key)) el.classList.add('adjudicated');

    const kind = f.payload?.kind || 'none';
    el.innerHTML = `
      <div class="fc-top">
        <span class="fc-id">${f.check_id}</span>
        <span class="sev-tag ${f.severity}">${f.severity}</span>
      </div>
      <div class="fc-scan">${f.scan_id}</div>
      <div class="fc-expl">${escapeHtml(f.explanation)}</div>
      <span class="fc-kind">payload: ${kind}</span>
      <div class="fc-actions">
        <button class="act-flyto" title="Send camera to this flag">Fly to</button>
        <button class="act-confirm ${this.decisions.get(key) === 'confirm' ? 'done' : ''}">Confirm</button>
        <button class="act-reject ${this.decisions.get(key) === 'reject' ? 'done' : ''}">Reject</button>
      </div>`;

    el.addEventListener('click', (e) => {
      if (e.target.closest('.fc-actions')) return;
      this._select(f, el);
    });
    el.querySelector('.act-flyto').addEventListener('click', () => this._select(f, el));
    el.querySelector('.act-confirm').addEventListener('click', () => this._decide(f, 'confirm', el));
    el.querySelector('.act-reject').addEventListener('click', () => this._decide(f, 'reject', el));
    return el;
  }

  _select(f, el) {
    this.selected = f;
    for (const c of this.listEl.children) c.classList?.remove('selected');
    el.classList.add('selected');
    this._renderDetail(f);
    this.onSelect?.(f);
  }

  _decide(f, decision, el) {
    const key = FlagList.key(f);
    this.decisions.set(key, decision);
    el.classList.add('adjudicated');
    el.querySelector('.act-confirm').classList.toggle('done', decision === 'confirm');
    el.querySelector('.act-reject').classList.toggle('done', decision === 'reject');
    this.onAdjudicate?.({ scan_id: f.scan_id, check_id: f.check_id, decision });
  }

  _renderDetail(f) {
    const loc = f.location?.world_mm
      ? `<div>location: ${f.location.world_mm.map((v) => v.toFixed(1)).join(', ')} mm</div>`
      : '';
    this.detailEl.innerHTML = `
      <div class="dp-title">${f.check_id}</div>
      <div>${escapeHtml(f.explanation)}</div>
      ${loc}`;
    renderExtra(this.detailEl, f.extra);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
}
