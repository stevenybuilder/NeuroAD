#!/usr/bin/env python
"""Generate the README demo screenshots by driving the running app headlessly.

Run in the sfg env with backend + frontend up:
    micromamba run -n sfg python scripts/demo_shots.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:5173"
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist", "--disable-dev-shm-usage"]
OUT = Path("docs/screenshots")


def click_flag(page, *needles):
    page.evaluate(
        """(needles) => {
            const c = [...document.querySelectorAll('.flag-card')]
                .find(el => needles.every(n => el.textContent.includes(n)));
            if (c) c.click();
        }""",
        list(needles),
    )


def wait_overlay(page, includes=None, excludes=None):
    page.wait_for_function(
        """([inc, exc]) => {
            const n = window.sfg.viewer.nv.volumes.map(v => v.name)
                .concat(window.sfg.viewer.nv.meshes.map(m => m.name || ''));
            return (!inc || n.some(x => x.includes(inc))) && (!exc || !n.some(x => x.includes(exc)));
        }""",
        arg=[includes, excludes], timeout=20000,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=ARGS)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("() => window.sfg && window.sfg.viewer.nv.volumes.length > 0", timeout=30000)

        # 1. Annotation gallery (every payload kind).
        page.evaluate("async () => { await window.sfg.loadGallery(); }")
        page.wait_for_function("() => window.sfg.annotations.meshes.length > 0", timeout=30000)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "01_gallery.png"))

        # Run the whole verification pass.
        page.evaluate("async () => { await window.sfg.runChecks(); }")
        page.wait_for_selector(".flag-card", timeout=60000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "02_cohort_flags.png"))

        # 3. Skull-strip: weak stripper over-inclusion (red whole-head mask).
        click_flag(page, "1.2.skull_strip", "payload: mask")
        wait_overlay(page, includes="weakstrip", excludes="reg-residual")
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "03_skullstrip_weak.png"))

        # 4. Skull-strip: SynthStrip reference brain surface (3D, clipped to
        #    reveal the green brain mesh inside the head volume render).
        click_flag(page, "1.2.skull_strip", "payload: mesh")
        page.wait_for_timeout(600)
        page.click('.chip[data-view="render"]')
        page.evaluate("() => window.sfg.viewer.setClip(0.0)")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "04_skullstrip_synthstrip.png"))
        page.evaluate("() => window.sfg.viewer.setClip(0.5)")
        page.click('.chip[data-view="multi"]')

        # 5. Intensity: cross-scanner histograms.
        click_flag(page, "1.3.intensity")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "05_intensity_confound.png"))

        # 6. Registration: residual mismatch heatmap.
        click_flag(page, "1.5.registration", "missing")
        wait_overlay(page, includes="reg-residual", excludes="weakstrip")
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "06_registration_mismatch.png"))

        # 7. Orientation: LR-flip laterality markers (3D render).
        click_flag(page, "1.1.orientation", "LRFLIP")
        page.wait_for_timeout(600)
        page.click('.chip[data-view="render"]')
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "07_orientation_lrflip.png"))

        browser.close()
    print(f"wrote demo screenshots to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
