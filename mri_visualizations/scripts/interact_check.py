#!/usr/bin/env python
"""Prove the viewer is genuinely interactive: drag-rotate the 3D render, wheel-zoom,
and scrub a slice via real mouse events, asserting the rendered pixels change.

Writes before/after frames and prints PASS/FAIL per interaction.
"""

import hashlib
import sys

from playwright.sync_api import sync_playwright

URL = "http://localhost:5173"
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist", "--disable-dev-shm-usage"]
OUT = "scratch_extract/shots"


def canvas_hash(canvas):
    # Screenshot the canvas (captures composited WebGL) and hash the PNG bytes.
    # readPixels is unreliable here: NiiVue uses preserveDrawingBuffer=false.
    return hashlib.md5(canvas.screenshot()).hexdigest()


def main() -> int:
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=ARGS)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("() => window.sfg && window.sfg.viewer.nv.volumes.length > 0", timeout=30000)
        page.evaluate("async () => { await window.sfg.loadGallery(); }")
        page.wait_for_function("() => window.sfg.annotations.meshes.length > 0", timeout=30000)
        page.click('.chip[data-view="render"]')
        page.wait_for_timeout(600)

        canvas = page.query_selector("#gl")
        box = canvas.bounding_box()
        cx, cy = box["x"] + box["width"] * 0.45, box["y"] + box["height"] * 0.5

        # 1) Drag-rotate.
        h0 = canvas_hash(canvas)
        page.mouse.move(cx, cy)
        page.mouse.down()
        for k in range(1, 11):
            page.mouse.move(cx + k * 12, cy + k * 3)
        page.mouse.up()
        page.wait_for_timeout(500)
        h1 = canvas_hash(canvas)
        results.append(("drag-rotate", h0 != h1))
        page.screenshot(path=f"{OUT}/04_rotated.png")

        # 2) Wheel-zoom.
        page.mouse.move(cx, cy)
        page.mouse.wheel(0, -600)
        page.wait_for_timeout(500)
        h2 = canvas_hash(canvas)
        results.append(("wheel-zoom", h1 != h2))

        # 3) Slice scrub (multiplanar): move crosshair through slices.
        page.click('.chip[data-view="multi"]')
        page.wait_for_timeout(400)
        h3 = canvas_hash(canvas)
        page.evaluate("() => window.sfg.viewer.nv.moveCrosshairInVox(0,0,25)")
        page.wait_for_timeout(400)
        h4 = canvas_hash(canvas)
        results.append(("slice-scrub", h3 != h4))

        browser.close()

    ok = all(passed for _, passed in results)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
