// The producer-agnostic annotation renderer.
//
// It renders a Flag purely by `payload.kind` and never asks who produced the
// flag - a deterministic check, an agent, or NeuroJEPA all land here identically.
// To add a new payload kind, add one branch; nothing else changes.

import { resourceUrl } from './api.js';

const to255 = (rgba) => rgba.map((c) => Math.round(c * 255));

function boxObjBlob(min, max) {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  const v = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const f = [
    [1, 2, 3], [1, 3, 4], [5, 6, 7], [5, 7, 8],
    [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
    [3, 4, 8], [3, 8, 7], [4, 1, 5], [4, 5, 8],
  ];
  const text =
    v.map((p) => `v ${p[0]} ${p[1]} ${p[2]}`).join('\n') + '\n' +
    f.map((t) => `f ${t[0]} ${t[1]} ${t[2]}`).join('\n') + '\n';
  return URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
}

export class Annotations {
  constructor(nv) {
    this.nv = nv;
    this.overlays = []; // NVImage refs added as overlays
    this.meshes = [];   // NVMesh refs
    this.labels = [];   // NVLabel3D refs
  }

  async render(flag) {
    const p = flag.payload || { kind: 'none' };
    switch (p.kind) {
      case 'mask':
        return this._mask(p);
      case 'heatmap':
        return this._heatmap(p);
      case 'mesh':
        return this._mesh(p);
      case 'point':
        return this._point(p);
      case 'points':
        return this._points(p);
      case 'bbox':
        return this._bbox(p);
      case 'none':
      default:
        return; // plaintext-only: nothing to draw
    }
  }

  async renderAll(flags) {
    this.clear();
    for (const f of flags) {
      // One malformed payload must not sink the rest of the annotations.
      try {
        await this.render(f);
      } catch (e) {
        console.warn(`annotation render failed for ${f.check_id}:`, e);
      }
    }
    this.nv.drawScene();
  }

  /** Isolate a single flag's annotation (used when a flag is selected). */
  async renderOne(flag) {
    this.clear();
    try {
      await this.render(flag);
    } catch (e) {
      console.warn(`annotation render failed for ${flag.check_id}:`, e);
    }
    this.nv.drawScene();
  }

  async _mask(p) {
    const img = await this.nv.addVolumeFromUrl({
      url: resourceUrl(p.resource),
      name: p.resource,
      colormap: p.colormap || 'red',
      opacity: p.opacity ?? 0.5,
      cal_min: 0.5,
      cal_max: 1.0,
      alphaThreshold: true, // background (below cal_min) stays transparent
      colorbarVisible: false,
    });
    this.overlays.push(img);
  }

  async _heatmap(p) {
    const img = await this.nv.addVolumeFromUrl({
      url: resourceUrl(p.resource),
      name: p.resource,
      colormap: p.colormap || 'warm',
      opacity: p.opacity ?? 0.6,
      cal_min: p.cal_min ?? undefined,
      cal_max: p.cal_max ?? undefined,
      alphaThreshold: true, // low residual stays transparent instead of tinting the skull
      colorbarVisible: false,
    });
    this.overlays.push(img);
  }

  async _mesh(p) {
    const mesh = await this.nv.addMeshFromUrl({
      url: resourceUrl(p.resource),
      name: p.resource,
      rgba255: to255(p.rgba || [1, 0.3, 0.3, 1]),
    });
    this.meshes.push(mesh);
  }

  _point(p) {
    const rgba = p.rgba || [1, 1, 0, 1];
    const label = this.nv.addLabel(
      p.text || '',
      {
        textColor: rgba,
        textScale: 0.6,
        lineWidth: 2,
        lineColor: rgba,
        bulletScale: 0.6,
        bulletColor: rgba,
      },
      p.coord_mm,
    );
    this.labels.push(label);
  }

  _points(p) {
    for (const m of p.markers || []) this._point(m);
  }

  async _bbox(p) {
    const url = boxObjBlob(p.min_mm, p.max_mm);
    const rgba = p.rgba || [0.2, 0.8, 1, 1];
    // NiiVue infers mesh format from the URL extension; a bare blob: URL has
    // none, so append a #.obj fragment (ignored by the blob fetch itself).
    const mesh = await this.nv.addMeshFromUrl({
      url: `${url}#bbox.obj`,
      name: 'bbox.obj',
      rgba255: [...to255(rgba).slice(0, 3), 45],
    });
    this.meshes.push(mesh);
    URL.revokeObjectURL(url);
    if (p.text) {
      const center = p.min_mm.map((v, i) => (v + p.max_mm[i]) / 2);
      this._point({ coord_mm: center, text: p.text, rgba });
    }
  }

  clear() {
    for (const o of this.overlays) {
      try { this.nv.removeVolume(o); } catch { /* already gone */ }
    }
    for (const m of this.meshes) {
      try { this.nv.removeMesh(m); } catch { /* already gone */ }
    }
    if (this.labels.length) {
      const mine = new Set(this.labels);
      this.nv.document.labels = this.nv.document.labels.filter((l) => !mine.has(l));
    }
    this.overlays = [];
    this.meshes = [];
    this.labels = [];
    this.nv.drawScene();
  }
}
