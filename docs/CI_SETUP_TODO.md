# TODO: enable GitHub Actions CI (manual step)

The CI workflow for this repo is **written and ready but not yet installed as an active
workflow**, because the machine that pushed the repo was signed in with a GitHub token that
lacks the `workflow` OAuth scope (GitHub refuses to accept pushes that add/modify files under
`.github/workflows/` without it). Everything else is pushed and current.

The exact workflow content is preserved in this repo at **`docs/ci-workflow.yml.txt`**.

## To turn it on, pick ONE of these:

**Option A — via GitHub web UI (no scope needed):**
1. On GitHub, click **Add file → Create new file**.
2. Name it `.github/workflows/ci.yml`.
3. Paste the contents of `docs/ci-workflow.yml.txt`.
4. Commit. Actions will start running on the next push/PR.

**Option B — from the terminal (grant the scope once):**
```bash
gh auth refresh -h github.com -s workflow    # authorize in the browser
mkdir -p .github/workflows
cp docs/ci-workflow.yml.txt .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Enable CI workflow"
git push
```

## What the CI does (why it's worth enabling)
On every push/PR it installs from a clean checkout on Python 3.11 + 3.12, runs the full
`pytest` suite, and smoke-tests both offline entrypoints (`neuroad demo` and
`neuroad reproduce-finding`) — guarding the "runs in one command, reproducible from a clean
clone" claim. The resulting green ✓ badge is a credibility signal for the *Depth & Execution*
judging dimension.
