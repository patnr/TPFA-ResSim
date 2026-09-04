# Changelog

Notable changes to `TPFA-ResSim`.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is semantic, but `0.x`, meaning that **minor bumps may break the API**.
Such changes are marked **BREAKING** below.

The package is not on PyPI; it is consumed straight from git, so downstream users
should pin a tag (or commit hash) and advance it deliberately.

## [Unreleased]

### Added

- **An adjoint model**, `TPFA_ResSim.tlm`, derived by hand: `linearize`
  recomputes a step of `time_stepper` (from the trajectory that `sim` returns)
  into a `Tape`, `adj_step` propagates a sensitivity back through it, and
  `adjoint` sweeps a whole trajectory, returning the gradient of an objective
  with respect to `S0`, `P0` and `log K` at the cost of about one `sim`. The
  other parameters, the controls' dependence on the state, and the discrete
  decisions (sub-step count, upwind directions) are held fixed; explicit scheme
  only. Verified against finite differences (`tests/test_tlm.py`); derivation
  and caveats in the module docstring. Illustrated by
  `examples/water_cut_gradient.py` and `examples/history_match_gradient.py`.
- **Cached preconditioning of the pressure solve**, on by default
  (`ResSim.cached_precond`): conjugate gradients preconditioned by an earlier
  step's factorization, refreshed on non-convergence, in place of a fresh
  factorization per step. Exact to the solver tolerance, so no recorded value
  changes; 2--6x faster where the saturation hardly moves (well test,
  depletion), 15--40% in a waterflood. `cached_precond=False` restores the
  direct solve, itself 1.5x faster from a symmetric ordering. Pinned by
  `tests/test_precond.py`.
- **A well model**: `peaceman_WI` computes Peaceman's well index from the grid,
  the (possibly anisotropic) permeability, the well radius and the skin.
  Assigned to the new `Wells.WI`, it makes `sim` record bottom-hole pressures
  in the new `wells.actual_bhp` (via the new `ResSim.bhp`) -- grid-independent
  to 0.2% (`tests/test_wells.py`), unlike the cell pressure that `sim` used to
  advertise as "BHP-like". Defaults to `None`, so no existing result changes.
- **BHP-controlled wells**, opt-in via the new `wells.bhp` (shaped like
  `rates`; `nan` entries stay rate-controlled, so the modes mix across wells
  and in time). Solved simultaneously with the pressure, not lagged: prescribing
  a rate-controlled run's `actual_bhp` reproduces it to machine precision. A
  BHP well also anchors the incompressible pressure equation, lifting both its
  pin and its balance requirement. No native mode switching, and no declared
  flow direction -- ref the docstring's warning.
- **Well paths**: `well_path` discretizes a polyline into one completion per
  traversed cell, with well indices (scaled by the traversed fraction) and a
  rate allocation. Under BHP control the completions share a `p_bh`, so the
  split is solved for; under rate control it is prescribed (the bordered
  system that would solve it is deliberately not implemented).
- `ResSim.well_controls(S, P, k)`: the feedback-control hook, replacing
  `dynamic_rate` (ref Changed). It returns `dict(rates=..., bhp=...)`, so an
  override governs the wells' *modes* as well as their rates, approximating
  the mode switching (e.g. a rate target with a BHP limit) that the model does
  not do natively -- lagged by one step. Worked examples in its docstring.
- **`ResSim.wells`**: the well configuration as one record (`dict`) per well,
  assigned to this attribute or passed to the constructor:

  ```python
  model = ResSim(
      Lx=1,
      Ly=1,
      Nx=64,
      Ny=64,
      wells=[
          dict(name="I1", path=[[0, 0], [0, 1]], rate=+1, rw=1e-3),
          dict(name="P1", xy=[1, 1], rate=-1),
      ],
  )
  ```

  `Wells.from_records` documents the keys and assembles the per-completion
  arrays, which remain the whole of the (writable) configuration. A
  hand-written config and its record equivalent run bit for bit alike
  (`tests/test_well_config.py`).
- **Well grouping**: `wells.group` maps each completion to its well and
  `wells.names` names them (set by the records above). Hence `wells.nWell`,
  `wells.rates_by_well`, and plot markers labelled by name (`plt_production`
  accepts `labels`). Deliberately no `bhp_by_well`: rates aggregate,
  pressures do not.
- `examples/well_control.py` (rate control, BHP control, and a rate target with
  a BHP limit) and `examples/well_path.py` illustrate the above.
- **`TPFA_ResSim.wells`**, a module of its own, holding the `Wells` dataclass
  and the free functions `peaceman_WI` and `well_path`. What couples the wells
  to the fluids or to the linear system (`assemble_wells`, `realize_bhp`,
  `bhp`, `well_controls`) stays in `__init__.py`, some 400 lines lighter.
- **`ResSim.cdarcy`**: Darcy's constant, whereby the model may be posed in
  *practical* (non-coherent) units -- metric (m, day, bar, mD, cP) being
  `0.008527`, ECLIPSE's `CDARCY`. Its docstring is the units story (the
  formula, the forced *areal* rate unit, a table of systems);
  `tests/test_units.py` pins it across four unit systems. The default `1`
  leaves every existing result unchanged. `examples/buildup.py` is re-posed in
  metric and read as a well test.

### Changed

- **BREAKING**: injectors and producers are **unified into a single set of
  wells**, removing every per-kind code path:
  - `inj_xy`/`prd_xy` -> `wells.xy`; `nInj`/`nPrd` -> `wells.nWell`.
  - `inj_rates`/`prd_rates` -> `wells.rates`, now **signed**: positive injects
    (water), negative produces. The transport step already discriminated by
    the sign of the source field, so nothing else needs to know a well's kind.
    The incompressible balance assertion is now that the rates sum to 0.
  - `wells.actual_rates` is a single, signed `(nComp, nSteps)` array, no
    longer a dict by kind -- likewise the rates of the control hook.
  - Plot markers are inferred from the sign of each well's rates (ref
    `Wells.signs`), with a new neutral marker for the undecided. The
    numbering is unchanged, being per sign. `well_scatter`'s `inj: bool`
    becomes `sgn: int` (+1/-1/0), which silently reinterprets an old
    positional `False` as `0`.
  - A lone well no longer needs a zero-rate partner of the other kind.
  - The regression values are unaffected; where a digest recorded production
    rates, they are negated back to positive.
- **BREAKING**: `wells.nWell` counts **wells**, not completions -- the latter
  being `wells.nComp` (forwarded by `ResSim.nComp`), the equations being
  assembled per completion. Since `nWell` only arrived with the unification
  above, no released version is affected.
- The examples all configure their wells by records now. Regression values
  unchanged.
- `_set_Q` is renamed `assemble_wells` and made public, with its partner
  `realize_bhp` (a few callers set up a pressure solve without `sim`). They
  write the source field `_Q` and a bundle `_wells_now` of per-well arrays and
  the BHP wells' contributions to `TPFA`; neither returns anything.
- `wells.actual_rates` and `wells.actual_bhp` are declared attributes, `None`
  until `sim` allocates them, and recorded from `time_stepper` rather than
  from the well physics.
- **BREAKING**: `dynamic_rate` is removed in favour of `well_controls`, a
  superset of it. Porting an override:
  `rates = super().dynamic_rate(S, k)` becomes
  `ctrl = super().well_controls(S, P, k)`, with `rates` now `ctrl["rates"]`,
  a single signed array (ref `tests/test_wells.py`). No shim: the hook had no
  known downstream overriders, and a rate-only hook cannot express a mode
  switch.
- **BREAKING**: `assemble_wells(S, k)` -> `assemble_wells(S, P, k)`, `P`
  being the pressure at the start of the step (`None` if there is none).
- **BREAKING**: `sim`'s initial-condition arguments are renamed `x0, p0` ->
  `S0, P0`, matching the `(SS, PP)` it returns.
- **BREAKING**: `TPFA` and `pressure_step` return the pressure *flat* (`Nxy`),
  like the saturation, rather than grid-shaped; their `p_prev` argument is
  renamed `P`. Callers that index the pressure in 2D must reshape.
- The **`struct-tools` dependency is dropped**. `NicePrint` is replaced by
  `AlignedRepr` (`TPFA_ResSim._repr`), whose `repr` summarizes big arrays and
  is also the `str`. The two `DotDict`s become a `Fluxes` named tuple (still
  `V.x`/`V.y`) and the plain `dict` `_wells_now`. NB: a downstream that
  imported `struct_tools` transitively from here (HistoryMatching does) must
  add it to its own requirements.

### Fixed

- **`Ny = 1` runs** used to raise `ValueError: offset array contains duplicate
  values`, the x- and y-neighbour diagonals coinciding at `±1`. `_spdiags` now
  sums coincident diagonals, so a row reproduces a column (`Nx=1`) to
  round-off (`tests/test_transport.py`). Plotting a 1D field is still not
  possible (`contourf` wants a `(2, 2)` array).
- The explicit scheme's sub-step count is kept off the round-off
  (`estimate_1CFL` shaves a relative `1e-9`): round-numbered set-ups put
  `dt * cfl1` exactly on an integer, where the linear solver's last bits
  decided the count, differently across platforms. Only
  `examples/buckley_leverett.py` sat there; its references now take the
  smaller count throughout (`tests/test_transport.py`).
- The $O(c_t)$ term of the **transport** equation, previously neglected, is
  now included (`ResSim.storage_rate`; ref the "Compressibility" section of
  the docs). A single-phase reservoir now stays single-phase, whereas
  previously an injector's cell accumulated water until it ran away, which
  limited `ct`. Saturations for `ct > 0` change accordingly (the compressible
  examples' references are updated); `ct = 0` is bit-for-bit unaffected.
- Scalar `K` (e.g. `ResSim(..., K=3.)`) broadcasts as documented, instead of
  raising `ValueError: cannot reshape array of size 2` (broken since
  `4643295`, v0.1.1).
- Two tolerances that were *absolute* are now **relative**, the magnitudes
  being a matter of the units (ref `cdarcy`): the rate-balance check of
  `time_stepper` and the upper-border nudge of `Grid2D.xy2sub`. Neither
  changes any existing result.

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
- `e3d4026`: A hook for state-dependent (feedback) well control:
  `dynamic_rate()`, since generalized to `well_controls()`.
- `d827ce8`, `70c19e5`: Plotting facilities (fields, streamlines, wells,
  animation) as a mixin.
- `9300beb`: Optional slight compressibility (`ct > 0`); the Matlab codes are
  strictly incompressible.
- `dac8634`: Type hints, checkable with `ty`.

[0.2.0]: https://github.com/patnr/TPFA-ResSim/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/patnr/TPFA-ResSim/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/patnr/TPFA-ResSim/releases/tag/v0.1.0
