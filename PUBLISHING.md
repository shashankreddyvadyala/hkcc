# Publishing hkcc to PyPI

The package is built and validated. The name `hkcc` is currently free on PyPI.

## 0. One-time edit
If your GitHub username isn't `svadyala`, update the URLs in `pyproject.toml`
(and the links in `README.md` / the website) before publishing.

## 1. Build (already done; rerun after any change)
```bash
pip install build twine
rm -rf dist
python -m build            # creates dist/hkcc-0.2.0.tar.gz and the .whl
twine check dist/*         # must say PASSED for both
```

## 2. Dry run on TestPyPI (recommended first)
Create an account at https://test.pypi.org and an API token (Account → API tokens).
```bash
twine upload --repository testpypi dist/*
# then verify it installs from TestPyPI in a clean venv:
python -m venv /tmp/t && /tmp/t/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ hkcc
/tmp/t/bin/python -c "import hkcc; print(hkcc.__version__)"
```

## 3. Publish to real PyPI
Create an account at https://pypi.org and an API token.
```bash
twine upload dist/*
```
Use `__token__` as the username and the `pypi-...` token as the password
(or put it in `~/.pypirc`). After this, `pip install hkcc` works for everyone,
and the PyPI link on the website goes live.

## 4. Releasing new versions
- Bump `version` in `pyproject.toml` (PyPI refuses re-uploading an existing version).
- Rebuild, `twine check`, upload.
- Tag it in git: `git tag v0.2.0 && git push --tags`.

## Optional install extras
```
pip install hkcc                 # core (numpy, scipy)
pip install "hkcc[documents]"    # + pypdf, python-docx
pip install "hkcc[embeddings]"   # + sentence-transformers
pip install "hkcc[all]"          # everything
```
