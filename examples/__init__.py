"""Runnable illustrations of the simulator.

Each is a plain, top-to-bottom script -- run as `python examples/buildup.py`, say --
whose docstring says what it shows, and whose figures are reproduced on its page here.

They double as regression tests: `tests/test_examples.py` runs them all
(plotting included) and compares the output with `tests/references.py`.

One panel per feature, drawn from their results:

![One panel per feature, from the examples](collage_features.png)

- `examples.quarter_five_spot` reproduces Fig. 6 of the reference paper -- this is
  what verifies our agreement with the Matlab codes -- and then varies it: the
  implicit transport scheme, and *scheduled* (time-varying) injection rates, which
  steer the water front.
- `examples.buckley_leverett` is the only *verification* among them: in 1D the
  saturation equation is exactly solvable (by the Welge tangent construction),
  so here the numerical profile is compared with the truth rather than with
  ourselves, and the error is shown to vanish under grid refinement.
- `examples.egg` is the one *validation* against an external simulator: the Egg
  model (a channelized, 12-well benchmark reservoir), flattened from 7 layers to
  one by vertical averaging, reproduces the water cuts and oil rates of its
  published 3D solution (ECLIPSE 100) to within 0.01 and about 5%. Also the second
  example in metric units, and the one with non-quadratic relative permeabilities
  (Corey exponents 3/4 with end-points, ref `TPFA_ResSim.fluids.Fluid`).
- `examples.inactive_cells`: an irregular reservoir on the rectangular grid --
  an outline and a sealing fault, cut out by `TPFA_ResSim.ResSim.active`.
- `examples.aquifer`: water beyond part of the boundary, feeding a lone producer
  -- a BHP-controlled "well" in the contact cells (`TPFA_ResSim.wells.aquifer_WI`),
  at constant pressure, or depleting (Fetkovich: a `well_controls` override).
- `examples.logo`: a smiley and a yin-yang -- outlines, a hole and a barrier cut
  out by `active`, an aquifer along the bottom -- for the picture alone.

These concern the *well model* (`TPFA_ResSim.wells.peaceman_WI`), i.e. the sub-grid
relation between a well and the (much larger) cell that holds it:

- `examples.well_control`: the two ways to control a well -- prescribing its rate and
  letting its pressure follow, or the reverse -- shown to be one model seen from
  either end. Also why the well model is needed at all: a well's *cell* pressure
  is a grid artefact, whereas the bottom-hole pressure derived from it is not.
  Its setting is a lone producer depleting a closed reservoir, whose transient
  and boundary-dominated regimes are seen in the drawdown.
- `examples.well_path`: a well completed along a polyline rather than in a single
  cell, and the two ways its rate then gets divided among the completions --
  statically (in proportion to the well index) or, under BHP control, solved for.

The next ones illustrate what slight compressibility (`TPFA_ResSim.ResSim.ct` > 0) brings:

- `examples.pressure_diffusion`: the pressure equation becomes parabolic, so that a
  change of rate propagates at *finite speed* (diffusivity `η = K λ / (φ ct)`),
  instead of being felt everywhere instantaneously. Also illustrates that the
  pressure level is now meaningful (anchored by `p0`), whereas the incompressible
  pressure is only defined up to a constant.
- `examples.buildup`: production *without* injection (impossible if incompressible),
  with the resulting material-balance decline, `dp̄/dt = -q / (ct Vp)`; then
  shutting the well in, and the ensuing pressure buildup. Monitor points far
  from the well respond late -- and keep declining after the shut-in, before
  turning around. Posed in metric units, and interpreted as a well test.
- `examples.voidage_replacement`: the only *two-phase* one of these -- waterflooding
  while injecting only half of what is produced (impossible if incompressible).
  The front then advances more slowly, and by a different pattern, since some of
  the oil is instead driven by expansion.

The last two illustrate the adjoint (`TPFA_ResSim.tlm`), i.e. gradients of an
objective wrt the initial state, the permeability field and the BHP controls, checked
against finite differences:

- `examples.water_cut_gradient`: the sensitivity of one producer's water cut, to the
  permeability field and to the producers' BHP schedule.
- `examples.history_match_gradient`: a few steepest-descent steps towards a
  synthetic truth.
"""
