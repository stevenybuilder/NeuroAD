#!/usr/bin/env python3
"""Generate a single self-contained player: scenes embedded as base64 data-URIs.
Double-click the output to present all scenes with zero server / zero sibling files."""
import base64, re, pathlib

DIR = pathlib.Path(__file__).parent
MASTER = (DIR / "neuroad_demo_animations.html").read_text(encoding="utf-8")

SCENES = [
    ("scene0_title.html",       "Title",      "Title / opener"),
    ("scene0b_validation.html", "Validation", "Validated in the field"),
    ("scene1_pipeline.html",    "Pipeline",   "Discovery pipeline"),
    ("scene4_probe.html",       "Probe",      "One probe, three questions"),
    ("scene3_claude.html",      "Claude",     "Orchestrated by Claude"),
]

def b64(name):
    raw = (DIR / "scenes" / name).read_text(encoding="utf-8").encode("utf-8")
    return base64.b64encode(raw).decode("ascii")

# Build the replacement SCENES array (label/hint + inline base64 data)
entries = []
for name, label, hint in SCENES:
    entries.append(
        '    { label:%r, hint:%r, data:"%s" }' % (label, hint, b64(name))
    )
scenes_js = "  var SCENES = [\n" + ",\n".join(entries) + "\n  ];"

out = MASTER

# 1) Replace the SCENES array (from 'var SCENES = [' up to the closing '];')
out = re.sub(r"  var SCENES = \[.*?\];", scenes_js, out, count=1, flags=re.S)

# 2) Swap the iframe source mechanism: file path -> data URI
out = out.replace(
    "frame.src = SCENES[i].file;",
    'frame.src = "data:text/html;charset=utf-8;base64," + SCENES[i].data;',
)

# 3) Update the "open direct" hint + title to reflect self-contained mode
out = out.replace(
    '<div class="hint">open direct: <kbd>scenes/…</kbd></div>',
    '<div class="hint">self-contained · no server needed</div>',
)
out = out.replace(
    "<title>NeuroAD — Demo Animations (Player)</title>",
    "<title>NeuroAD — Demo Animations (Standalone)</title>",
)

dest = DIR / "neuroad_demo_animations_standalone.html"
dest.write_text(out, encoding="utf-8")
kb = len(out.encode("utf-8")) / 1024
print(f"wrote {dest.name}  ({kb:.0f} KB, {len(SCENES)} scenes embedded)")
