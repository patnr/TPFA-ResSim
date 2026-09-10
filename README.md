# MiniRes

[![GitHub CI](https://github.com/patnr/MiniRes/actions/workflows/tests.yml/badge.svg)](https://github.com/patnr/MiniRes/actions)

A 2D, two-phase, immiscible reservoir simulator
using TPFA (two-point flux approximation).
[**Documentation**](https://patnr.github.io/MiniRes/minires.html).

- **small**: all of its physics fit in `core.py`'s 300 lines of code
- **capable**: two-phase, slight compressibility, BHP control, irregular outlines and faults (inactive cells), aquifers –
  but 2D uniform grid, immiscible, isothermal, and only simple well models.
- **adjoint** model included; verified against finite differences
- **python**: easy to demo in a web browser via Colab or WASM
- **fast**: comparable to JutulDarcy (but no waiting on JIT startup!) at equal accuracy on 2D two-phase cases
- **reliable**: reproduces the numbers of the [Matlab code (2007)](http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf) from NTNU/Sintef by Jørg E. Aarnes, Tore Gimse, and Knut–Andreas Lie
- extensively **tested** and **documented**

![The Egg model: permeability, pressure, oil saturation, and the adjoint sensitivity of a producer's water cut](collage.png)

## Examples

The examples double as regression tests.
Each has a page in the [documentation](https://patnr.github.io/MiniRes/examples.html).

![One panel per feature, from the examples](collage_features.png)

## Used by

Please let me know (or make a PR) if you use this in your work,
and I will add it to this list.

- [History matching tutorial](https://github.com/patnr/HistoryMatching)

## Installation

The package is not on PyPI (yet -- a release is planned), so install it from git.
Since `0.x` minor bumps may break the API (ref `CHANGELOG.md`),
pin a tag (or a commit hash) and advance it deliberately.

Requires Python `>=3.12`.


```sh
pip install "minires @ git+https://github.com/patnr/MiniRes.git@v0.2.0"
```

or, with [uv](https://docs.astral.sh/uv/),

```sh
uv add "minires @ git+https://github.com/patnr/MiniRes.git@v0.2.0"
```
## Contributions

To also get the examples and tests, clone instead, and install in editable mode:

```sh
git clone https://github.com/patnr/MiniRes.git
cd MiniRes
uv sync  # or: pip install -e .
uv run pytest
uv run ruff check
```
