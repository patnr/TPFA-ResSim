"""Tests of the explicit transport scheme's sub-stepping.

The upwind scheme (`ResSim.saturation_step_upwind`) splits each `dt` into
`nT = ceil(dt / dt_CFL)` sub-steps, and the *result depends on `nT`*: more
sub-steps means a smaller sub-`dt`, hence more numerical diffusion. That
ceiling is therefore a discontinuity in the output -- one that a `dt` landing
exactly *on* a multiple of the CFL limit sits astride. Round-numbered set-ups
readily do so (ref `examples/buckley_leverett.py`, whose 1D column puts
`1/CFL` at exactly `3 N`), and the decision would then be made by the last
bits of `spsolve`'s fluxes, i.e. differently on different platforms -- which
is what the epsilon in `saturation_step_upwind` exists to prevent.

The set-up below is that column: unit rate into unit pore volume over `N`
cells, with `dt` chosen for exactly 6 sub-steps.
"""

import numpy as np

from TPFA_ResSim import ResSim

N = 200
kws: dict = dict(Lx=1, Ly=1, Nx=1, Ny=N,
                 wells=[dict(xy=[0, 0], rate=+1), dict(xy=[0, 1], rate=-1)])

dt = 6 / (3*N)  # exactly 6 sub-steps' worth
nSteps = 20
S0 = np.zeros(N)


def test_on_the_cfl_boundary_takes_the_lower_count():
    """`dt` *on* the boundary must sub-step as `dt` a hair *below* it does.

    Both want 6 sub-steps; the two runs may then differ only by the hair in
    `dt` itself. Without the epsilon they differ by a thousand times that,
    the round-off having bought the first run some 7-sub-step steps.
    """
    S_on, _ = ResSim(**kws).sim(dt, nSteps, S0, pbar=False)
    S_below, _ = ResSim(**kws).sim(dt * (1 - 1e-6), nSteps, S0, pbar=False)
    assert abs(S_on - S_below).max() < 1e-3


def test_the_epsilon_does_not_swallow_a_real_sub_step():
    """A `dt` genuinely *over* the boundary must still get its 7th sub-step.

    I.e. the epsilon ($10^{-9}$, relative) is far narrower than any `dt` one
    would choose on purpose -- so it absorbs round-off, and nothing else.
    """
    S_on, _ = ResSim(**kws).sim(dt, nSteps, S0, pbar=False)
    S_above, _ = ResSim(**kws).sim(dt * (1 + 1e-6), nSteps, S0, pbar=False)
    assert abs(S_on - S_above).max() > 1e-3


def test_sub_stepping_is_insensitive_to_round_off():
    """Perturbing `K` by a relative $10^{-14}$ must not flip the sub-step count.

    The linear solve carries such a perturbation into the fluxes, and thence
    into the CFL estimate -- which is precisely the noise that differs between
    platforms and library versions. The saturations may then move by round-off,
    but not by the (much larger) step of a changed `nT`.
    """
    S1, _ = ResSim(**kws).sim(dt, nSteps, S0, pbar=False)
    S2, _ = ResSim(K=1 + 1e-14, **kws).sim(dt, nSteps, S0, pbar=False)
    assert abs(S1 - S2).max() < 1e-9
