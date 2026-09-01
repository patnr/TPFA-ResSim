# Changelog

Notable changes to `TPFA-ResSim`.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is semantic, but `0.x`, meaning that **minor bumps may break the API**.
Such changes are marked **BREAKING** below.

The package is not on PyPI; it is consumed straight from git, so downstream users
should pin a tag (or commit hash) and advance it deliberately.

## [Unreleased]

### Added

- A **well model**: `ResSim.peaceman_WI` computes Peaceman's well index from the
  grid, the (possibly anisotropic) permeability, the well radius and the skin;
  assigning it to the new `inj_WI`/`prd_WI` attributes makes `sim` record the
  implied bottom-hole pressures in `actual_bhp` (via the new `ResSim.bhp`).
  Unlike the cell pressure -- which `sim`'s docstring could only advertise as a
  "bottom-hole pressure-*like* observable", and which varies by some 45% over a
  16² -> 64² refinement -- this is grid-independent, matching the analytic
  (Dietz) drawdown to within 0.2% on every grid (`tests/test_wells.py`).
  The well index is also a diagnostic in its own right: it defaults to `None`
  (whereupon `actual_bhp` is `nan`), and a rate-controlled well is unaffected by
  it, so adding one changes no existing result.
- **BHP-controlled wells**, opt-in via the new `inj_bhp`/`prd_bhp` attributes
  (shaped like the rates, so likewise time-varying; `nan` entries stay
  rate-controlled, hence the two modes may be mixed across wells and in time).
  The rate is solved for *simultaneously* with the pressure -- `TPFA` puts
  $WI λ_t$ on its diagonal and $WI λ_t p_{bh}$ on its right-hand side -- rather
  than lagged by a step: prescribing the `actual_bhp` of a rate-controlled run
  reproduces it to machine precision (`tests/test_wells.py`). The realized rates
  are reported in `actual_rates`, so `inj_rates` may be left `None`.
  A BHP well also *anchors* the incompressible (`ct == 0`) pressure equation,
  which is otherwise a pure-Neumann problem: the arbitrary pin at the SW corner
  (ref article p. 13) is then skipped, the absolute pressure level becomes
  meaningful, and injection need no longer balance production.
  There is no switching of control modes; a producer that would flow backwards
  raises rather than doing so silently.
- **Well paths**: `ResSim.well_path` walks a polyline through the grid and
  returns the traversed cells, their well indices (each scaled by how much of
  its cell the path actually crosses), and the resulting split of the well's
  rate. Several completions act as one well by being superimposed in
  `assemble_wells`,
  which already worked; what is new is the discretization, the well indices, and
  the rate allocation. Under BHP control the completions simply share a
  `p_bh` -- all a wellbore does, absent the gravity and friction that a 2D areal
  model has no room for -- so their split is then *solved for* rather than
  prescribed. A rate-controlled multi-cell well whose split is solved for would
  need `p_bh` as an extra unknown, i.e. a bordered linear system, and is
  deliberately not implemented.
- `examples/well_control.py` and `examples/well_path.py` illustrate the above
  (and, like the other examples, double as regression tests).

### Changed

- `_set_Q` is renamed `assemble_wells` -- and made public, together with its
  partner, `realize_bhp`: `examples/heterogeneous.py` and a couple of the tests
  already called the private names, to set up a pressure solve without `sim`.
  What they compute for the step now travels by one channel, in two places: the
  source field, `_Q`, and a bundle, `_wells_now`, holding the per-well arrays
  (`inds`, `rates`, `p_bh`, `WI_lam` -- each a `dict` by well kind) and the
  per-cell terms that the BHP wells contribute to `TPFA` (`bhp_diag`,
  `bhp_rhs`). Neither method returns anything.
- `actual_rates` and `actual_bhp` are now declared (and documented) attributes,
  `None` until `sim` allocates them -- rather than springing into existence
  mid-simulation, guarded by `hasattr`. The recording itself moved out of the
  well physics (`assemble_wells`/`realize_bhp`) and into `time_stepper`.
- **BREAKING**: `sim`'s initial-condition arguments are renamed `x0, p0` ->
  `S0, P0`, matching the `(SS, PP)` trajectories it returns. Only `p0` was
  likely passed by keyword; `S0` is positional in practice.
- **BREAKING**: `TPFA` and `pressure_step` now return the pressure *flat*
  (`Nxy`), i.e. shaped like the saturation state, rather than grid-shaped --
  reshaping only internally, for the flux extraction. Callers that plot it (or
  `reshape`/`ravel` it) are unaffected; those that index it in 2D must reshape.
  Their `p_prev` argument is likewise renamed `P`, the pressure now being a
  state variable that the call advances in time.

### Fixed

- The `O(ct)` term of the **transport** equation, previously neglected, is now
  included: `ResSim.storage_rate` is the volume rate going into storage, and the
  saturation steps charge it to the phases in proportion to their saturation
  (the split that makes the water and oil equations sum to the pressure
  equation). Consequently a single-phase reservoir now stays single-phase --
  whereas previously, depleting a water-filled one drained the saturation by the
  voidage, and an injector's cell accumulated water (`~ ct*dp_well`) until, via
  the mobility of the excess, it ran away. The limit this imposed on `ct` is
  thereby lifted (for the explicit scheme, which is monotone; the implicit
  Newton solver can still stray outside `[0, 1]` far beyond the model's
  premise); what remains is the model's own `ct*dp << 1` premise.
  Saturations for `ct > 0` change accordingly (the compressible examples'
  reference values are updated); `ct = 0` is bit-for-bit unaffected.
- Scalar `K` (e.g. `ResSim(..., K=3.)`) now broadcasts as documented, instead of
  raising `ValueError: cannot reshape array of size 2`. Broken since `4643295`
  ("Reshape K in setter", v0.1.1), which used `np.full_like(self.shape, ...)`
  -- i.e. filling an array shaped like the *tuple* `(Nx, Ny)`, not like the grid.

## [0.2.0] -- 2026-08-27

Compressibility, and a general tooling refresh.

### Added

- **Slight compressibility**, opt-in via the new `ct` attribute (`9300beb`).
  With `ct > 0` the pressure equation becomes parabolic (backward Euler), so
  pressure propagates at finite speed, the absolute pressure level becomes
  meaningful (anchored by `p0`), and injection need no longer balance production
  -- enabling e.g. primary depletion. The corresponding `O(ct)` term in the
  *transport* equation was deliberately neglected in this release (documented,
  with the resulting limit on `ct`); ref Unreleased, above.
- `examples/`: runnable illustrations that double as the regression test suite
  (`c1873dc`). `heterogeneous.py` and `quarter_five_spot.py` are the former
  `tests/test_fig1.py` and `test_fig6.py`; `rate_scheduling.py`,
  `pressure_diffusion.py`, `depletion.py`, `buildup.py` and
  `voidage_replacement.py` are new.
- Type hints throughout, checkable with `ty` (a dev dependency; not wired into CI)
  (`dac8634`).

### Changed

- **BREAKING**: `sim()` now returns the `(S, P)` trajectories rather than just
  the saturation `S`, and accepts an optional initial pressure `p0` (`9300beb`).
  Callers must unpack: `S, P = model.sim(...)`.
- **BREAKING**: minimum Python raised to 3.12 (from 3.9); CI matrix is now
  3.12--3.14 (`d21e48a`).
- Packaging/dev environment migrated from poetry to uv (`d21e48a`).
- Linting switched from flakeheaven to ruff (`1760b00`).
- Minimum matplotlib raised to 3.8 (`60e1813`).

### Fixed

- Figures now display when running as a plain script or under IPython, and
  `anim()` works on matplotlib >= 3.10, where `ContourSet` is itself an artist
  (`60e1813`).

## [0.1.1] -- 2023-10-24

Mostly an API-ergonomics release: well configuration, plotting and the grid
became attributes/methods of the model rather than free functions and setup
calls.

### Added

- Time-dependent well rates: `inj_rates`/`prd_rates` are reshaped to
  `(nWell, nTime)`, with singletons broadcast over time (`988b47e`).
- `dynamic_rate(S, k)`: an override point for rates that depend on the current
  state (e.g. shutting wells on water breakthrough), plus the `actual_rates`
  record of what the wells really did (`e3d4026`).
- Simulations with no flow at all (no injectors/producers, or zero rates) now
  run, without warnings (`f0ff50e`, `ebc9235`).
- Convenience properties `nInj`, `nPrd` (`34323da`, `adc8953`), and a `name`
  attribute (`f3ce235`).
- Plotting: grid overlay in `plt_field()` (`14f45dc`), a `locator` style key
  (`c2254ee`), `finalize` in the plotters (`0525c45`), well-marker sizing via
  forwarded kwargs (`f2afbaa`), and a clearer bullseye argmax indicator
  (`0301231`).

### Changed

- **BREAKING**: `config_wells()` removed. Well specs are plain attributes
  (`inj_xy`, `inj_rates`, `prd_xy`, `prd_rates`) whose setters do the
  normalization -- snapping positions to the nearest grid node and reshaping
  rates (`18588dd`, `55d6f35`). `K` is likewise broadcast in its setter
  (`4643295`).
- **BREAKING**: `recurse()` renamed to `sim()` (`89b1c0e`).
- **BREAKING**: `prod`/`nProd` renamed to `prd`/`nPrd` throughout (`adc8953`).
- **BREAKING**: `ResSim` is now a dataclass, changing the constructor signature
  (`708f756`, `f665312`).
- **BREAKING**: plotting functions became methods of the inherited `Plot2D`
  mixin (`70c19e5`), plot coordinates are absolute rather than relative
  (`d8369d1`), and `plt_field()` changed its `colorbar`/`wells` defaults
  (`60594a6`).
- **BREAKING**: `M` renamed to `Nxy` (`095ec3f`), `.grid` renamed to `.domain`
  (`c13a8b7`), and `Q` made private (`34a57a6`).
- **BREAKING**: out-of-bounds coordinate conversions raise instead of being
  silently clamped (`1c3754c`); rates are asserted non-negative (`f405043`).
  Assertions generally gained messages (`6f8db6a`).
- Lists are accepted wherever arrays are (`6ca6a4c`).

## [0.1.0] -- 2023-03-29

Initial Python translation of the Matlab codes of Aarnes, Gimse & Lie (2007):
the TPFA pressure solver, both the explicit-upwind and implicit
(Newton--Raphson) saturation steppers, the grid utilities, the plotting
facilities, and the reproductions of the paper's Figs. 1 and 6 that verify
agreement with the Matlab output. See the deviations below.

## Deviations from the Matlab codes

Structural differences from the original, as opposed to changes over time.
The Python code still reproduces the Matlab output (up to linear-solver and
randomness differences), as verified by `examples/quarter_five_spot.py`.

- `83293bc`: Converted from 3D to 2D for simplicity.
- `a9fcc49`: Index ordering is C-major (numpy standard), not F-major.
- `7543f57`: Vectors are "numpy-thonic", in using 1d arrays, not (2d) columns.
- `cade315`: Several linear solvers suggested.
- `f33c571`: OOP -- so that ensembles of independent models can be forecast.
- `55ce732`: Facilities for working on the grid.
- `e0d12b0`, `988b47e`: Convenient well config, with rates that may vary in
  time. Total injection must equal total production only in the incompressible
  case; see `ct` below.
- `e3d4026`: `dynamic_rate()` hook for state-dependent well control.
- `d827ce8`, `70c19e5`: Plotting facilities (fields, streamlines, wells,
  animation) as a mixin.
- `9300beb`: Optional slight compressibility (`ct > 0`); the Matlab codes are
  strictly incompressible.
- `dac8634`: Type hints, checkable with `ty`.

[0.2.0]: https://github.com/patnr/TPFA-ResSim/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/patnr/TPFA-ResSim/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/patnr/TPFA-ResSim/releases/tag/v0.1.0
