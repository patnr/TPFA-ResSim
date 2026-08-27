# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 2D, two-phase (water/oil), immiscible toy reservoir simulator using two-point flux approximation (TPFA), translated from the Matlab codes of Aarnes, Gimse & Lie (kept in `matlab_codes/`, paper in `refs/aarnes2007introduction.pdf`). Incompressible by default (`ct = 0`), with optional slight compressibility (`ct > 0`, backward-Euler accumulation term in the pressure equation, and the corresponding storage term in the transport equation, charged to the phases in proportion to their saturation — ref `ResSim.storage_rate`). The Python code is verified to reproduce the Matlab output — the regression values in doctests and tests encode this, so do not "fix" numeric expected values unless the physics intentionally changed.

## Downstream usage

The main consumer is [patnr/HistoryMatching](https://github.com/patnr/HistoryMatching) (locally at `~/P/HistoryMatching`), an ensemble history-matching tutorial that uses `ResSim` as its forward model. Consequences for changes here:

- It pins this repo by **git commit hash** in its `requirements.txt` (with a commented-out `-e` local-path line for development), so changes here don't break it until the pin is advanced — but breaking changes should be flagged so the pin advance can be coordinated. E.g. its notebooks call `model.sim(dt, nTime, wsat0, pbar=False)` expecting a single array; v0.2.0's `(S, P)` tuple return will require updating them.
- Its de facto API surface is wider than `sim()`: well-config attributes (`inj_xy`, `inj_rates`, ..., incl. the singleton time-broadcast behaviour), grid conversions (`xy2ind`, `ind2xy`), `shape`/`Nxy`, and the plotting mixin (`plt_field`, `plt_production`, `anim`). Notably it **mutates `TPFA_ResSim.plotting.styles`** to register its own field styles — treat that dict and the `plt_field(style=...)` mechanism as public API.
- Ensemble forecasting is why `ResSim` is OOP with per-instance state (see below), and why `sim` accepts `pbar=False`: many instances/runs, no shared mutable state, quiet loops.

## Commands

Dev environment is managed with uv (`uv sync`), but a plain `pip install -e .` also works. Note: `mise.toml` sets `UV_PROJECT_ENVIRONMENT` so the venv lives at `~/.cache/venvs/TPFA-ResSim`, not in-project — if mise isn't active in your shell, export that variable before running uv to avoid creating a duplicate `.venv`.

Plain `uv sync` installs the default (`dev`) group, i.e. everything. CI instead installs
only what it runs, via the narrower `test` and `docs` groups that `dev` includes
(`uv sync --no-default-groups --group test`) — which keeps the interactive tools out of
the CI environment, notably `pdbpp`, whose shadowing of the stdlib `pdb` that pytest and
`doctest` import can break *collection* on an interpreter it dislikes. The subsequent step
is then `uv run --no-sync`, since `uv run` would otherwise re-sync the default group and
reinstall what the sync step just excluded.

- **Run all tests**: `uv run pytest` (no further args). `addopts` in pyproject.toml makes this run `tests/` **plus doctests** in all `TPFA_ResSim` modules (`--doctest-modules`). Doctests are part of the test suite.
- **Run a single test**: override addopts, e.g. `uv run pytest -o addopts="" "tests/test_examples.py::test_example[quarter_five_spot]"`
- **Lint**: `uv run ruff check` (config in pyproject.toml; max line length 88; deliberately permissive about operator/array alignment — preserve the aligned formatting style used in the code).
- **Type check**: `uv run ty check` (clean, except that ty cannot verify `**dict` splatting into a typed signature (used in examples. Not run in CI — CI lints with `uvx ruff check` only.
- **Docs**: `uv run pdoc --math -o docs/ ./TPFA_ResSim` (published to GitHub Pages by `.github/workflows/docs.yml`). Docstrings are pdoc-flavoured markdown with LaTeX math; `TPFA_ResSim/README.md` is included into the package docstring via `.. include::`.

The supported Python range is whatever `requires-python` in pyproject.toml says (currently `>=3.12`); the floor tracks Colab's Python so the package installs there without re-installs. CI tests 3.12–3.14 on ubuntu + macos.

## Architecture

The package is three files; the physics lives in `TPFA_ResSim/__init__.py`:

- **`ResSim`** (`__init__.py`) is a dataclass composed by multiple inheritance: `ResSim(NicePrint, Grid2D, Plot2D)`. OOP is used (rather than passing dicts) so ensemble forecasting (as in HistoryMatching) can hold independent instances whose parameters don't influence each other.
  - `sim(dt, nSteps, x0, p0=None)` is the entry point, returning the `(S, P)` saturation and pressure trajectories: it loops `time_stepper`, which per step solves pressure (`TPFA()` → sparse direct solve; elliptic if `ct == 0`, else parabolic backward-Euler) then transports saturation with either the explicit upwind scheme (`saturation_step_upwind`, sub-steps from a CFL estimate) or the implicit Newton–Raphson scheme (`saturation_step_implicit`, halves sub-dt until convergence).
  - Method names reference listings in the paper (e.g. `TPFA()` = Listing 1, `RelPerm()` = Listing 6).
  - **`__setattr__` does normalization magic**: assigning `inj_xy`/`prd_xy` snaps well positions to the nearest grid node; `inj_rates`/`prd_rates` are reshaped to `(nWell, nTime)`; scalar `K` is broadcast to shape `(2, Nx, Ny)`. Rates must be positive; when `ct == 0`, total injection must additionally equal total production at every time index (asserted in `time_stepper` — otherwise mass imbalance would silently leak in the SW corner). With `ct > 0` the imbalance is absorbed by storage (enabling e.g. primary depletion via zero-rate injectors).
  - `dynamic_rate(S, k)` is a designed override point (patch/subclass) for e.g. shutting wells based on saturation.
- **`Grid2D`** (`grid.py`): rectangular grid geometry and the coordinate/index conversions (`xy2ind`, `sub2xy`, ...). Index ordering is **C-major (numpy default), unlike the Matlab original**: `x` is the first axis, so printing a field matrix shows x as the row index, whereas plots show x left→right, y bottom→top. This ordering is hard-coded in the simulator via `ravel`/`reshape`.
- **`Plot2D`** (`plotting.py`): plotting mixin (fields, streamlines, well markers).

`examples/` holds the runnable illustrations (incl. the figures of `collage.jpg`); they
double as the regression tests. Each is a plain top-to-bottom script whose only concession
to the harness is a final `__digest__` dict of the values to be checked, and the guarded
`if __name__ == "__main__": show()`. `tests/test_examples.py` runs them all with `runpy`
(so `show()` is skipped, but the plotting *is* exercised) and compares a fingerprint of
`__digest__` with `tests/references.py` (regenerate that table with
`uv run python tests/test_examples.py`, but only if the change is intended).
`tests/test_compressible.py` is different: structural/physics properties, no figures.
