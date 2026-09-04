"""Tests of the explicit transport scheme's sub-stepping, and of the 1D layouts.

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


def test_row_equals_column():
    """A 1D row of cells (`Ny=1`) must reproduce the 1D column (`Nx=1`).

    The two are the same problem, transposed. But `TPFA` and `upwind_diff`
    place the x- and y-neighbours at offsets `±Ny` and `±1`, which coincide
    when `Ny == 1` -- and `scipy.sparse` rejects duplicate offsets, so
    `ResSim._spdiags` has to merge them. Heterogeneous `K`, compressibility
    and a BHP well exercise every diagonal that gets merged.
    """
    N = 40
    K = np.linspace(1, 3, N)
    fluid: dict = dict(swc=.2, sor=.2, vw=1., vo=2., ct=.1)
    col = ResSim(Lx=1, Ly=1, Nx=1, Ny=N, K=K.reshape(1, N), **fluid,
                 wells=[dict(xy=[0, 0], rate=+1), dict(xy=[0, 1], bhp=0., rw=.01)])
    row = ResSim(Lx=1, Ly=1, Nx=N, Ny=1, K=K.reshape(N, 1), **fluid,
                 wells=[dict(xy=[0, 0], rate=+1), dict(xy=[1, 0], bhp=0., rw=.01)])
    S_col, P_col = col.sim(.05, 10, np.zeros(N), np.zeros(N), pbar=False)
    S_row, P_row = row.sim(.05, 10, np.zeros(N), np.zeros(N), pbar=False)
    assert S_col.shape == S_row.shape == (11, N)
    assert np.allclose(S_col, S_row, atol=1e-12)
    assert np.allclose(P_col, P_row, atol=1e-9)
    assert np.allclose(col.wells.actual_rates, row.wells.actual_rates, atol=1e-12)
    # And the water has actually travelled (i.e. the flow is not trivial).
    assert .3 < S_row[-1].mean() < .7
