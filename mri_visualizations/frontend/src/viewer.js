// NiiVue wrapper: owns the base volume, the view layout, and the camera.
// Annotation rendering lives in annotations.js so the viewer stays agnostic to
// what a flag is.

import { Niivue, SHOW_RENDER } from '@niivue/niivue';
import { volumeUrl } from './api.js';

export class Viewer {
  constructor(canvas, hud) {
    this.hud = hud;
    this.nv = new Niivue({
      backColor: [0.03, 0.04, 0.06, 1],
      show3Dcrosshair: true,
      crosshairColor: [0.3, 0.55, 1, 1],
      crosshairWidth: 1,
      isColorbar: false,
      textHeight: 0.04,
      isOrientCube: true,
    });
    this.nv.attachToCanvas(canvas);
    // Show the 3D render tile alongside the three ortho planes in multiplanar.
    this.nv.opts.multiplanarShowRender = SHOW_RENDER.ALWAYS;
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);
    this.nv.onLocationChange = (loc) => this._updateHud(loc);
    this.currentScan = null;
    this.currentModality = null;
  }

  /** Load a base volume, replacing whatever volumes are loaded. Overlays and
   *  annotations are the caller's responsibility to re-apply afterwards. */
  async loadScan(scanId, modality) {
    this.currentScan = scanId;
    this.currentModality = modality;
    await this.nv.loadVolumes([
      { url: volumeUrl(scanId, modality), name: `${modality}.nii.gz`, colormap: 'gray' },
    ]);
    if (this.hud) this.hud.textContent = `${scanId}  ·  ${modality}`;
  }

  setView(mode) {
    const nv = this.nv;
    if (mode === 'multi') nv.setSliceType(nv.sliceTypeMultiplanar);
    else if (mode === 'render') nv.setSliceType(nv.sliceTypeRender);
    else if (mode === 'axial') nv.setSliceType(nv.sliceTypeAxial);
  }

  setClip(depth) {
    // depth in [-0.5..0.5]; 0.5 disables the clip plane (whole volume shown).
    if (depth >= 0.49) this.nv.setClipPlane([2, 0, 0]); // pushed out of view
    else this.nv.setClipPlane([depth, 270, 0]);
  }

  /** Send the crosshair (and thus all ortho views) to a world/mm coordinate. */
  flyTo(worldMm) {
    if (!worldMm || !this.nv.volumes.length) return;
    const frac = this.nv.mm2frac(worldMm);
    this.nv.scene.crosshairPos = frac;
    this.nv.createOnLocationChange();
    this.nv.drawScene();
  }

  _updateHud(loc) {
    if (!this.hud) return;
    const mm = loc?.mm ? [...loc.mm] : null;
    const val = loc?.values?.[0]?.value;
    const parts = [];
    if (this.currentScan) parts.push(this.currentScan);
    if (mm) parts.push(`mm ${mm.slice(0, 3).map((v) => v.toFixed(1)).join(', ')}`);
    if (val !== undefined) parts.push(`val ${Number(val).toFixed(1)}`);
    this.hud.textContent = parts.join('   ·   ');
  }
}
