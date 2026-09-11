# MiniRes

<img src="logo.png" alt="The MiniRes logo" align="right" width="300"/>

A simple petroleum reservoir simulator
using TPFA (two-point flux approximation).
[**Documentation**](https://patnr.github.io/MiniRes/minires.html).

- **Small**: all of its physics fit in `core.py`'s 400 lines of code.
- **Capable**: two-phase, slight compressibility, BHP control, well paths, irregular outlines and faults (inactive cells), aquifers –
  **but** 2D uniform grid, immiscible, isothermal, and simple well models and operation.
- **Adjoint** model included; verified against finite differences.
- **Python**: easy to demo in a web browser via
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patnr/MiniRes/blob/main/notebooks/colab.ipynb) (backend: Google)
  or [![WASM](https://img.shields.io/static/v1?label=WASM&message=by%20marimo&logo=webassembly&color=654ff0)](https://patnr.github.io/MiniRes/wasm/) (no backend!).
- **Fast**: similar to [JutulDarcy's](https://github.com/sintefmath/JutulDarcy.jl) (but no JIT startup/wait) at equal accuracy on 2D two-phase cases of size $100$ – $10^5$.
- **Reliable**: reproduces the numbers of the [Matlab code (2007)](http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf) from NTNU/Sintef by Jørg E. Aarnes, Tore Gimse, and Knut–Andreas Lie.
  Further validated against Buckley–Leverett's
  analytic solution, ECLIPSE's numbers on the Egg model
  and JutulDarcy's on quarter five-spot, SPE-10, and Egg.
- **Tested** extensively: [![GitHub CI](https://github.com/patnr/MiniRes/actions/workflows/tests.yml/badge.svg)](https://github.com/patnr/MiniRes/actions),
  with many [examples](https://patnr.github.io/MiniRes/examples.html) doubling as regression tests.

![The Egg model: permeability, pressure, oil saturation, and the adjoint sensitivity of a producer's water cut](collage.png)

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
