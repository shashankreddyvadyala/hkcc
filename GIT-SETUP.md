# Pushing this repo to GitHub + hosting the site

The username is already wired in as `shashankreddyvadyala`. Everything is committed
(`git log`). You just point it at a new GitHub repo and push.

## 1. Create an empty repo on GitHub
https://github.com/new → name it `hkcc` → leave it empty (no README/license).

## 2. Push
```bash
git remote add origin https://github.com/shashankreddyvadyala/hkcc.git
git push -u origin main
```

## 3. Turn on the website (GitHub Pages)
Repo → Settings → Pages → Build and deployment → Source: **GitHub Actions**.
The `Deploy site` workflow publishes `docs/` on every push. Your site goes live at:

    https://shashankreddyvadyala.github.io/hkcc/

(Alternative without Actions: Source → "Deploy from a branch" → `main` / `/docs`.)

## 4. CI
The `CI` workflow runs the 14-test suite on every push and PR (Python 3.9 / 3.11 / 3.12).
The README badge turns green once it passes.

## 5. Publishing to PyPI (when ready)
By hand: see PUBLISHING.md. Or automatically:
1. Repo → Settings → Secrets and variables → Actions → add `PYPI_API_TOKEN`
   (token from https://pypi.org/manage/account/token/).
2. Create a GitHub Release (tag `v0.2.0` → Publish).
3. The `Publish to PyPI` workflow builds and uploads automatically.

## Layout
```
src/hkcc/   library    tests/   14 tests    examples/  quickstart + RAG harness
benchmarks/ benchmarks paper/   tex+pdf     docs/      website (Pages)
.github/    CI, Pages deploy, PyPI publish workflows
```
