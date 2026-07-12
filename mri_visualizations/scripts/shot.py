#!/usr/bin/env python
"""Headless screenshot harness for visual confirmation of the viewer.

Drives the running web app with Playwright (software WebGL2 via SwiftShader),
performs an optional scripted action, waits for NiiVue to settle, and writes a
PNG. Also surfaces any browser console errors so a broken build is loud.

Usage:
    python scripts/shot.py <action> <out.png> [scan_id]

Actions:
    boot     - default scan, multiplanar
    gallery  - load the annotation gallery (all payload kinds)
    render   - gallery + switch to 3D render view
    scan     - load a specific scan_id (3rd arg), multiplanar
"""

import sys

from playwright.sync_api import sync_playwright

URL = "http://localhost:5173"
ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--disable-dev-shm-usage",
]


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "boot"
    out = sys.argv[2] if len(sys.argv) > 2 else "shot.png"
    scan_id = sys.argv[3] if len(sys.argv) > 3 else None

    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=ARGS)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(URL, wait_until="networkidle")
        # Wait for boot: base volume loaded.
        page.wait_for_function("() => window.sfg && window.sfg.viewer.nv.volumes.length > 0", timeout=30000)

        if action in ("gallery", "render"):
            page.evaluate("async () => { await window.sfg.loadGallery(); }")
            page.wait_for_function(
                "() => window.sfg.annotations.meshes.length > 0 && window.sfg.annotations.overlays.length > 0",
                timeout=30000,
            )
        if action == "render":
            page.evaluate("() => window.sfg.viewer.setView('render')")
        if action == "scan" and scan_id:
            page.evaluate("async (id) => { await window.sfg.loadScan(id); }", scan_id)
        if action == "checks":
            page.evaluate("async () => { await window.sfg.runChecks(); }")
            page.wait_for_selector(".flag-card", timeout=30000)
            # Optionally select a specific flag (3rd arg = index) to fly to it.
            if scan_id is not None:
                page.evaluate(
                    "(i) => document.querySelectorAll('.flag-card')[i]?.click()", int(scan_id)
                )
            page.wait_for_timeout(800)

        page.wait_for_timeout(1200)  # let NiiVue finish drawing
        page.screenshot(path=out)
        browser.close()

    if errors:
        print("BROWSER ERRORS:")
        for e in errors[:20]:
            print("  -", e)
    print(f"wrote {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
