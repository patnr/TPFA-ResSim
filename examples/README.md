They double as regression tests: `tests/test_examples.py` runs them all
(plotting included) and compares the output with `tests/references.py`.

- `heterogeneous.py` and `quarter_five_spot.py` reproduce Figs. 1 and 6 of the
  reference paper. The latter is what verifies our agreement with the Matlab codes.
- `rate_scheduling.py` steers the water front using time-varying injection rates.

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
