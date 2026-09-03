They double as regression tests: `tests/test_examples.py` runs them all
(plotting included) and compares the output with `tests/references.py`.

- `heterogeneous.py` and `quarter_five_spot.py` reproduce Figs. 1 and 6 of the
  reference paper. The latter is what verifies our agreement with the Matlab codes.
- `rate_scheduling.py` steers the water front using time-varying injection rates.
- `buckley_leverett.py` is the only *verification* among them: in 1D the
  saturation equation is exactly solvable (by the Welge tangent construction),
  so here the numerical profile is compared with the truth rather than with
  ourselves, and the error is shown to vanish under grid refinement.

These concern the *well model* (`TPFA_ResSim.wells.peaceman_WI`), i.e. the sub-grid
relation between a well and the (much larger) cell that holds it:

- `well_control.py`: the two ways to control a well -- prescribing its rate and
  letting its pressure follow, or the reverse -- shown to be one model seen from
  either end. Also why the well model is needed at all: a well's *cell* pressure
  is a grid artefact, whereas the bottom-hole pressure derived from it is not.
- `well_path.py`: a well completed along a polyline rather than in a single
  cell, and the two ways its rate then gets divided among the completions --
  statically (in proportion to the well index) or, under BHP control, solved for.

The remaining ones illustrate what slight compressibility (`ct > 0`) brings:

- `pressure_diffusion.py`: the pressure equation becomes parabolic, so that a
  change of rate propagates at *finite speed* (diffusivity `η = K λ / (φ ct)`),
  instead of being felt everywhere instantaneously. Also illustrates that the
  pressure level is now meaningful (anchored by `p0`), whereas the incompressible
  pressure is only defined up to a constant.
- `depletion.py`: production *without* injection (impossible if incompressible),
  its transient and boundary-dominated regimes, and the resulting material-balance
  decline, `dp̄/dt = -q / (ct Vp)`.
- `buildup.py`: shutting in a well, and the ensuing pressure buildup. Monitor
  points far from the well respond late -- and keep declining after the shut-in,
  before turning around.
- `voidage_replacement.py`: the only *two-phase* one of these -- waterflooding
  while injecting only half of what is produced (impossible if incompressible).
  The front then advances more slowly, and by a different pattern, since some of
  the oil is instead driven by expansion.
