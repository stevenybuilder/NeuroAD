# NeuroAD Discovery Engine — one-liners a reviewer can trust.
#
#   make install     editable install with the dev (pytest) extra
#   make test        run the full pytest suite
#   make demo        SURVIVOR-vs-KILL punchline on the synthetic harness (offline)
#   make reproduce   regenerate the ~0.93 PCA-10 scanner-leakage AUC from the
#                    checked-in fixture (backend WAVE-2 wires `reproduce-finding`)
#
# Everything runs offline: `anthropic` is an optional [claude] extra, and every
# Claude call degrades to a deterministic template when no API key is set.
#
# Override the interpreter if `python` isn't on PATH, e.g.:
#   make demo PYTHON=.venv/bin/python

PYTHON ?= python

# Run the module form so the CLI works from a clean clone without an install.
export PYTHONPATH := src

.PHONY: install test demo reproduce

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m neuroad.cli demo

reproduce:
	$(PYTHON) -m neuroad.cli reproduce-finding
