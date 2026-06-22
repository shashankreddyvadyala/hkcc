# Pushing this repo to GitHub + hosting the site

Everything is committed already (run `git log` to see). You just need to point it
at a GitHub repo and push.

## 1. Create an empty repo on GitHub
Go to https://github.com/new, name it `hkcc`, leave it empty (no README/license).
If your username is not `svadyala`, do a find-and-replace first:

```bash
grep -rl svadyala . --exclude-dir=.git | xargs sed -i 's/svadyala/YOUR_USERNAME/g'
git commit -am "Set GitHub username"
```

## 2. Push
```bash
git remote add origin https://github.com/YOUR_USERNAME/hkcc.git
git push -u origin main
```

## 3. Turn on the website (GitHub Pages)
Repo → Settings → Pages → Build and deployment → Source: **GitHub Actions**.
The `Deploy site` workflow then publishes `docs/` automatically on every push.
Your site goes live at:  https://YOUR_USERNAME.github.io/hkcc/

(Alternative without Actions: Source → "Deploy from a branch" → `main` / `/docs`.)

## 4. CI
The `CI` workflow runs the test suite on every push and pull request
(Python 3.9 / 3.11 / 3.12). The badge in the README turns green once it passes.

## 5. Publishing to PyPI (when ready)
Either run it by hand (see PUBLISHING.md), or use the automated workflow:
1. Repo → Settings → Secrets and variables → Actions → add `PYPI_API_TOKEN`
   (your `pypi-...` token from https://pypi.org/manage/account/token/).
2. Create a GitHub Release (Releases → Draft a new release → tag `v0.2.0` → Publish).
3. The `Publish to PyPI` workflow builds and uploads automatically.

## What's in here
```
src/hkcc/        the library
tests/           14 tests
examples/        quickstart + RAG validation harness
benchmarks/      synthetic + multimodal benchmarks
paper/           paper source (.tex), figures script, and PDF
docs/            the project website (served by Pages)
.github/         CI, Pages deploy, and PyPI publish workflows
```
