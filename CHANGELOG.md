# Changelog

Notable changes to `TPFA-ResSim`.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is semantic, but `0.x`, meaning that **minor bumps may break the API**.
Such changes are marked **BREAKING** below.

The package is not on PyPI; it is consumed straight from git, so downstream users
should pin a tag (or commit hash) and advance it deliberately.

## [Unreleased]

### Added

- A **well model**: `peaceman_WI` computes Peaceman's well index from the
  grid, the (possibly anisotropic) permeability, the well radius and the skin;
  assigning it to the new `Wells.WI` attribute (`(nComp,)`, `nan` entries
  allowed, for wells without a well model) makes `sim` record the
  implied bottom-hole pressures in the new `wells.actual_bhp`
  (via the new `ResSim.bhp`).
  Unlike the cell pressure -- which `sim`'s docstring could only advertise as a
  "bottom-hole pressure-*like* observable", and which varies by some 45% over a
  16² -> 64² refinement -- this is grid-independent, matching the analytic
  (Dietz) drawdown to within 0.2% on every grid (`tests/test_wells.py`).
  The well index is also a diagnostic in its own right: it defaults to `None`
  (whereupon `wells.actual_bhp` is `nan`), and a rate-controlled well is
  unaffected by it, so adding one changes no existing result.
- **BHP-controlled wells**, opt-in via the new `wells.bhp` attribute
  (shaped like the rates, so likewise time-varying; `nan` entries stay
  rate-controlled, hence the two modes may be mixed across wells and in time).
  The rate is solved for *simultaneously* with the pressure -- `TPFA` puts
  $WI λ_t$ on its diagonal and $WI λ_t p_{bh}$ on its right-hand side -- rather
  than lagged by a step: prescribing the `wells.actual_bhp` of a rate-controlled
  run reproduces it to machine precision (`tests/test_wells.py`). The realized rates
  are reported in `wells.actual_rates`, so `wells.rates` may be left `None`.
  A BHP well also *anchors* the incompressible (`ct == 0`) pressure equation,
  which is otherwise a pure-Neumann problem: the arbitrary pin at the SW corner
  (ref article p. 13) is then skipped, the absolute pressure level becomes
  meaningful, and injection need no longer balance production.
  There is no switching of control modes, nor a declared flow direction: a BHP
  well flows whichever way `p_bh` vs. its cell pressure dictates, an inflow
  injecting water like any other -- with a `UserWarning` if that direction
  flips mid-`sim`, such a reversal being more often a mistake than an intent.
- **Well paths**: `well_path` walks a polyline through the grid and
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
- `ResSim.well_controls(S, P, k)`: the feedback-control hook, replacing
  `dynamic_rate` (ref Changed, below). It returns `dict(rates=..., bhp=...)`
  (each a `(nComp,)` array),
  and is handed the pressure as well as the saturation, so an override governs
  the wells' *modes* as well as their rates -- which is what it takes to
  approximate the mode *switching* that the model does not do natively (ref
  `wells.bhp`): e.g. rate control with a BHP limit, wherein a producer holds its
  rate only until that would draw it below some `p_min` (worked examples in its
  docstring). Being judged from the previous step's pressure, the switch lags:
  the limit is breached for the one step in which it comes to bind, by less the
  shorter `dt` is (`tests/test_wells.py`).
- **`ResSim.wells`**: the well configuration -- one record (`dict`) per well,
  assigned to this new attribute, or straight to the constructor. Which is the
  convenient way to set the wells up:

  ```python
  model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64, wells=[
      dict(name="I1", path=[[0, 0], [0, 1]], rate=+1, rw=1e-3),
      dict(name="P1", xy=[1, 1], rate=-1),
  ])
  ```

  A record may hold a position (`xy`, one or several) or a well *path*
  (`path`), a `rate` and/or a `bhp` (a scalar, or a schedule over time), the
  wellbore radius `rw` (and `skin`) or the well index `WI` itself, and a
  `name`. Whence the helper computes what the model runs on: it discretizes a
  path into completions (`well_path`), computes their well indices
  (`peaceman_WI`), apportions the well's rate among them, shares out its BHP,
  broadcasts the constants against the schedules, and fills in the `0`/`nan`
  conventions of the specs it assembles.
  Assigning it is what applies it -- `__setattr__` forwards to
  `Wells.from_records`, in keeping with the normalization it already does for
  `K` -- and the arrays it assembles are the whole of the configuration: the
  records themselves are read, not retained, so nothing can fall out of step
  with a subsequent edit of the arrays (which remain assignable, ensemble
  methods perturbing `wells.xy` being unaffected). A hand-written config and
  its record equivalent produce the same run, bit for bit
  (`tests/test_well_config.py`).
- **Well grouping**, so that the reporting may speak of wells while the model
  solves for completions: `wells.group` maps each completion to its well, and
  `wells.names` names them (`wells` sets both; a `path` well's completions are
  thereby recognizably one well). Hence `nComp` -- the number of completions,
  i.e. the rows of `wells.xy`, `wells.actual_rates`, ... -- alongside
  `wells.nWell`, which counts *wells*; `wells.rates_by_well`, which sums
  `wells.actual_rates` per well; and plot markers labelled by name rather than
  by index (`plt_production` accepts `labels` to match).
  There is deliberately no `bhp_by_well`: rates aggregate, pressures do not.
- `examples/well_control.py` and `examples/well_path.py` illustrate the above
  (and, like the other examples, double as regression tests). The former's
  "modes" figure now contrasts all three: rate control, BHP control, and the
  rate target with a BHP limit -- whose curve traces the first until the limit
  binds, and the second thereafter.

- **`TPFA_ResSim.wells`**, a module of its own -- alongside `grid` and
  `plotting` -- holding everything that depends on the wells and the geometry
  alone: the `Wells` dataclass (the per-completion arrays, their
  normalization, the grouping, `rates_by_well`, `signs`, `at_time`) and the
  two free functions `peaceman_WI` and `well_path`, which take the model for
  its grid and `K`. What couples the wells to the *fluids* (`RelPerm`) or to
  the linear system (`_Q`, `_wells_now`) stays with the simulator --
  `assemble_wells`, `realize_bhp`, `bhp`, `well_controls` -- so the dependency
  runs one way, as it does for `Grid2D` and `Plot2D`.
  Some 400 lines lighter, `__init__.py` is left to the physics.
- **`ResSim.cdarcy`**: Darcy's constant, whereby the model may be posed in
  *practical* (non-coherent) units -- metric, say: m, day, bar, mD, cP. The
  equations otherwise contain no conversion factors (any *coherent* system,
  SI included, already works as-is), and this one constant is the whole of what
  a unit system amounts to here. It enters at just two sites, both of them
  Darcy's law: the transmissibilities of `TPFA` and the well index of
  `peaceman_WI`. Its docstring is where the units question is documented --
  the formula $C = u_k u_p u_t / (u_μ u_L^2)$ in terms of the units' SI sizes,
  the *forced* rate unit $u_q = u_L^2/u_t$ (an **areal** rate, the model being
  2D areal, so a rate per unit thickness), a table of systems and their $C$
  (metric `0.008527`, i.e. ECLIPSE's `CDARCY`; field-like `0.006328`; lab
  `3.6`), and the warning that $C$ neither names nor determines the units:
  it is one scalar constraining five choices, and going from metres to feet
  means re-expressing the *rates* too.
  `tests/test_units.py` poses one reservoir in each of those four systems and
  demands they agree -- which pins the constant's value, and the rate-unit
  exponent, as no regression value could; two deliberately-wrong variants
  confirm the check is not vacuous.
  The default is `1`, so every existing result, doctest and reference value is
  bit-for-bit unaffected. `examples/buildup.py` is re-posed in metric, and
  interpreted as a real well test would be: the plateau of its semilog
  derivative $dp/d\ln t$ recovers the 100 mD that went in, to within 4% (the
  shortfall being the time step). Nothing reads the units: its axis labels say
  "[bar]" because the script says so.
### Changed

- **BREAKING**: injectors and producers are **unified into a single set of
  wells**, removing every per-kind code path (the `("inj", +1)/("prd", -1)`
  loops, the dicts-by-kind, the paired attributes):
  - `inj_xy`/`prd_xy` -> `wells.xy`; `nInj`/`nPrd` -> `wells.nWell`.
  - `inj_rates`/`prd_rates` -> `wells.rates`, now **signed**: positive injects
    (water), negative produces (at the cell's fractional flow). This is the
    whole trick: the transport step already discriminated by the sign of the
    assembled source field, so with the sign in the spec, nothing anywhere
    needs to know a well's kind. The positivity assertion is gone; the
    incompressible balance assertion is now that the rates sum to 0, and
    `assemble_wells` asserts that a *rate*-controlled well has a finite rate
    (`nan` being tolerated only as a BHP-controlled well's placeholder).
  - `wells.actual_rates` is a single `(nComp, nSteps)` array (signed), no
    longer a dict by kind -- likewise the rates handed out by the hook (ref
    the `dynamic_rate` -> `well_controls` entry below).
  - Plot markers are inferred from the sign of each well's rates: the spec,
    `wells.rates`, falling back on the realized `wells.actual_rates` for the wells it
    leaves undecided (as pure BHP control does); only the still-undecided get
    the new neutral marker. Their *numbering* is unchanged, being per sign, so
    the producers are still numbered (and coloured) as in `plt_production`.
    `well_scatter`'s `inj: bool` parameter becomes `sgn: int` (+1/-1/0) --
    which, `bool` being an `int`, silently reinterprets an old positional
    `False` (producer) as `0` (neutral) rather than raising.
  - A lone well no longer needs a zero-rate partner of the other kind
    (e.g. `examples/depletion.py` sheds its idle injector).
  - The regression values are all unaffected (the physics is unchanged);
    where an example's digest recorded production rates, they are negated
    back to positive to keep `tests/references.py` identical.

- **BREAKING**: `wells.nWell` counts **wells**, not completions -- the count of
  those (the rows of `wells.xy`, `wells.rates`, `wells.actual_rates`, ...) being
  `wells.nComp`, which `ResSim.nComp` forwards, the equations being assembled
  per completion. The two coincide unless some well has several completions
  (ref `wells.group`). Since `nWell` itself only arrived with the unification
  above (from `nInj`/`nPrd`), no released version is affected.
- The examples all configure their wells by records now
  (`examples/well_path.py` most tellingly: its three cases no longer assemble
  the completions' arrays by `vstack`/`append` by hand). The regression values
  are unchanged.
- `_set_Q` is renamed `assemble_wells` -- and made public, together with its
  partner, `realize_bhp`: `examples/heterogeneous.py` and a couple of the tests
  already called the private names, to set up a pressure solve without `sim`.
  What they compute for the step now travels by one channel, in two places: the
  source field, `_Q`, and a bundle, `_wells_now`, holding the per-well arrays
  (`inds`, `rates`, `p_bh`, `WI_lam`) and the
  per-cell terms that the BHP wells contribute to `TPFA` (`bhp_diag`,
  `bhp_rhs`). Neither method returns anything.
- `wells.actual_rates` and `wells.actual_bhp` are now declared (and documented)
  attributes, `None` until `sim` allocates them -- rather than springing into existence
  mid-simulation, guarded by `hasattr`. The recording itself moved out of the
  well physics (`assemble_wells`/`realize_bhp`) and into `time_stepper`.
- **BREAKING**: `dynamic_rate` is removed in favour of `well_controls`, which
  is a superset of it: the rates are one of the two controls it returns.
  Porting an override means unwrapping that dict --
  `rates = super().dynamic_rate(S, k)` becomes
  `ctrl = super().well_controls(S, P, k)`, and `rates` becomes `ctrl["rates"]`,
  a single signed array over all wells
  -- as `tests/test_wells.py` illustrates. Keeping it as a shim was considered,
  but the hook had no known downstream overriders, and a rate-only hook cannot
  express a control-mode switch.
- **BREAKING**: `assemble_wells(S, k)` -> `assemble_wells(S, P, k)`, with `P`
  the pressure at the *start* of the step, which it passes on to
  `well_controls`. This affects only the direct callers that set up a pressure
  solve without `sim` (`examples/heterogeneous.py` and two tests, all updated);
  pass `None` if there is no pressure to offer.
- **BREAKING**: `sim`'s initial-condition arguments are renamed `x0, p0` ->
  `S0, P0`, matching the `(SS, PP)` trajectories it returns. Only `p0` was
  likely passed by keyword; `S0` is positional in practice.
- **BREAKING**: `TPFA` and `pressure_step` now return the pressure *flat*
  (`Nxy`), i.e. shaped like the saturation state, rather than grid-shaped --
  reshaping only internally, for the flux extraction. Callers that plot it (or
  `reshape`/`ravel` it) are unaffected; those that index it in 2D must reshape.
  Their `p_prev` argument is likewise renamed `P`, the pressure now being a
  state variable that the call advances in time.
- The **`struct-tools` dependency is dropped** -- only two of its facilities
  were used. `NicePrint` is replaced by `AlignedRepr`, a mixin of some 15 lines
  in the new (private) `TPFA_ResSim._repr`: the `repr` keeps its shape, but now
  *summarizes* big arrays (`K`, `por`) rather than dumping them in full, while
  `str` is no longer an alternative rendering (the bulleted one of
  `NicePrint.__str__`) -- it is simply the `repr`. And the two `DotDict`s
  become plain structs: the face fluxes are a `Fluxes` named tuple, which keeps
  `V.x`/`V.y` but *types* the `V` parameter of `upwind_diff`, `storage_rate`,
  `estimate_1CFL` and the two `saturation_step_*` (as well as the return of
  `TPFA` and `pressure_step`), whereas the internal `_wells_now` bundle is a
  `dict`, so its fields are keyed (`_wells_now["rates"]`, ...) rather than
  attributes. NB: a downstream that imported `struct_tools` *transitively* from
  here (HistoryMatching does) must add it to its own requirements.

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
- Two tolerances that were *absolute*, and so presumed values of order 1, are
  now **relative** -- which they must be, the magnitudes being a matter of the
  units (ref `ResSim.cdarcy`, above): the rate-balance check of `time_stepper` (an
  `np.isclose(sum, 0)`, i.e. `atol=1e-8`, which large rates could trip) and
  the upper-border nudge of `Grid2D.xy2sub` (`Lx - 1e-8`, which very large
  domains could round away). Neither changes any existing result.

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
