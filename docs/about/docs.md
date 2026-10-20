# Building these docs

This website is generated from the `docs/` folder of the repository with
[MkDocs](https://www.mkdocs.org) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme; the API
reference is read from the docstrings of the package by
[mkdocstrings](https://mkdocstrings.github.io).

## Build and browse

```bash
pip install -e ".[docs]"
mkdocs serve          # live preview at http://127.0.0.1:8000, rebuilt on every save
mkdocs build          # static site in site/
```

The built site works without a web server: open `site/index.html` in a
browser (search included). The published site at
<https://sacridini.github.io/landschaft/> is built the same way by GitHub
Actions (`.github/workflows/docs.yml`) with each release; *Actions > Docs >
Run workflow* publishes it again from `main` between releases.

## Figures and examples

- `python tools/make_doc_figures.py` redraws every figure of the tutorials
  from the example data (`docs/assets/figures`).
- `python tools/check_docs.py` runs every Python example of the pages, in
  order, so that the documentation never shows code that does not work;
  `python tools/check_docs.py --api` runs the examples of every docstring.
- `python tools/make_api_index.py` rewrites the overview of the API reference
  from the entries of its pages.

## Writing

Pages are Markdown with the Material extensions (admonitions, tabs, cards,
code annotations). API pages contain one `::: landschaft.function` directive
per function; the parameter tables, return values and examples come from the
numpy-style docstrings, so improving a docstring improves the website.
