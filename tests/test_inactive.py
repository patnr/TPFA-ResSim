"""Tests of inactive cells (`ResSim.active`).

Structural properties, no figures: an all-active mask changes nothing; a
rectangle of active cells embedded in a larger grid reproduces the smaller
grid; inactive cells are inert (closed faces, frozen state, no NaNs even at
zero porosity); an unbroken line of them seals; disconnected regions and a
well in an inactive cell are refused; the plots mask them out.
"""

import copy

import numpy as np
import pytest

from minires import ResSim

rng = np.random.default_rng(3)
dt, nSteps = .02, 8


def pair(xy_inj, xy_prd, q=1.0):
    return [dict(xy=xy_inj, rate=q), dict(xy=xy_prd, rate=-q)]


def run(model, S0=None, P0=None):
    S0 = np.zeros(model.Nxy) if S0 is None else S0
    return model.sim(dt, nSteps, S0, P0, pbar=False)


def test_all_active_is_the_default_and_changes_nothing():
    kws: dict = dict(Nx=10, Ny=8, K=np.exp(rng.standard_normal((10, 8))), ct=.1,
               wells=pair([0, 0], [1, 1], .6))
    ref = ResSim(**kws)
    assert ref.active.shape == ref.shape and ref.active.all()
    assert ref.active.dtype == bool and ref._pin == 0
    model = ResSim(**kws, active=np.ones((10, 8)))  # non-bool input is fine
    P0 = rng.random(ref.Nxy)
    for a, b in zip(run(ref, P0=P0), run(model, P0=P0)):
        assert np.array_equal(a, b)
    # The mask (and what is derived from it) survives copying
    clone = copy.deepcopy(model)
    assert np.array_equal(clone.active, model.active) and clone._pin == 0


@pytest.mark.parametrize("ct", [0, .05])
def test_embedded_rectangle_reproduces_the_smaller_grid(ct):
    """The physics of an active rectangle within a larger grid are those of
    that rectangle as a grid of its own (same cells, same pin)."""
    nx, ny, ox, oy = 9, 7, 3, 5  # the small grid, and its offset in the big one
    Nx, Ny = 16, 15
    K = np.exp(rng.standard_normal((2, nx, ny)))
    wells = pair([.5, .5], [nx - .5, ny - .5], .8)  # NB: `Lx = nx` ⇒ xy are cell indices
    small = ResSim(Lx=nx, Ly=ny, Nx=nx, Ny=ny, K=K, ct=ct, wells=wells)
    active = np.zeros((Nx, Ny), bool)
    active[ox:ox + nx, oy:oy + ny] = True
    assert not active[0, 0], "the pin must move for this to test anything"
    Kbig = np.ones((2, Nx, Ny))
    Kbig[:, active] = K.reshape(2, -1)
    big = ResSim(Lx=Nx, Ly=Ny, Nx=Nx, Ny=Ny, K=Kbig, ct=ct, active=active,
                 wells=pair([ox + .5, oy + .5], [ox + nx - .5, oy + ny - .5], .8))
    P0 = rng.random(small.Nxy)
    P0big = np.zeros(big.Nxy)
    P0big[active.ravel()] = P0
    SS, PP = run(small, P0=P0)
    SSb, PPb = run(big, P0=P0big)
    act = active.ravel()
    assert np.allclose(SSb[:, act], SS, rtol=0, atol=1e-12)
    assert np.allclose(PPb[:, act], PP, rtol=0, atol=1e-9)


@pytest.mark.parametrize("ct", [0, .1])
def test_inactive_cells_are_inert(ct):
    """Zero flux across their faces, no storage, and a state carried through
    unchanged -- also when their porosity is set to `0`, as is natural."""
    model = ResSim(Nx=14, Ny=12, ct=ct, wells=pair([0, 0], [1, 1]))
    active = np.ones(model.shape, bool)
    active[4:9, 3:8] = False
    active[12, 1] = False  # a lone one, too
    model.active = active
    por = np.ones(model.shape)
    por[~active] = 0
    model.por = por
    inact = ~active.ravel()
    S0, P0 = rng.random((2, model.Nxy))
    S0[~inact] = 0  # so that no water is produced before breakthrough
    SS, PP = run(model, S0, P0)
    assert np.isfinite(SS).all() and np.isfinite(PP).all()
    assert np.array_equal(SS[:, inact], np.tile(S0[inact], (nSteps + 1, 1)))
    assert np.array_equal(PP[:, inact], np.tile(P0[inact], (nSteps + 1, 1)))
    # The faces of the inactive cells: closed
    model.assemble_wells(SS[-1], PP[-1], 0)
    _, V = model.pressure_step(SS[-1], PP[-1], dt)
    fx = ~(active[:-1, :] & active[1:, :])  # interior x-faces touching one
    fy = ~(active[:, :-1] & active[:, 1:])
    assert not V.x[1:-1, :][fx].any() and not V.y[:, 1:-1][fy].any()
    assert not model.storage_rate(V)[inact].any()
    assert np.isinf(model.pore_volume()[inact]).all()
    # Meanwhile, the active cells do flow, and (if incompressible) conserve water
    assert abs(V.x).max() > 0
    if ct == 0:
        pv = model.pore_volume()[~inact]
        injected = dt * nSteps * 1.0  # no breakthrough this early
        assert np.isclose((pv * (SS[-1] - S0)[~inact]).sum(), injected)


def test_a_staircase_fault_seals():
    """An unbroken diagonal staircase of inactive cells blocks the flow (fluxes
    pass through faces only): the water must go around its tip."""
    model = ResSim(Nx=20, Ny=20, wells=pair([.05, .05], [.95, .05]))
    X, Y = model.mesh
    x_fault = .5 + .3 * (Y - .5)
    fault = (abs(X - x_fault) <= model.hx / 2 + 1e-9) & (Y < .7)  # from the bottom up
    model.active = ~fault
    SS, PP = model.sim(.02, 20, np.zeros(model.Nxy), pbar=False)
    beside = abs(X - x_fault) <= 1.5 * model.hx  # the columns hugging the fault
    below = (Y < .3) & ~fault  # ... near the wells, far from the tip
    left, right = beside & below & (X < x_fault), beside & below & (X > x_fault)
    assert SS[-1][left.ravel()].min() > .5, "the left side is flooded"
    # (what little is there has come around the tip, by numerical diffusion)
    assert SS[-1][right.ravel()].max() < 1e-9, "yet none of it crossed"
    # Break the line: it does cross
    active = ~fault
    active[np.flatnonzero(fault[:, 1])[0], 1] = True  # a gap, in row 1
    model.active = active
    SS, PP = model.sim(.02, 20, np.zeros(model.Nxy), pbar=False)
    assert SS[-1][right.ravel()].max() > .1


@pytest.mark.parametrize("precond", [False, True])
def test_disconnected_regions(precond):
    """A fault right across the reservoir makes two reservoirs. Only the first
    is pinned, so (if incompressible) the other must balance its own rates:
    else its system is singular -- which the solver does not notice by itself
    (no exception, just pressures of 1e14), but the residual check does."""
    active = np.ones((16, 10), bool)
    active[7, :] = False
    kws: dict = dict(Nx=16, Ny=10, active=active, cached_precond=precond)
    right = (np.arange(16) >= 7)[:, None].repeat(10, 1).ravel()
    # Each region balanced: fine, and each floods on its own
    model = ResSim(**kws, wells=pair([0, 0], [.4, 1], 1) + pair([1, 0], [1, 1], .3))
    SS, PP = run(model)
    assert SS[-1, ~right].sum() > SS[-1, right].sum() > 0 and abs(PP).max() < 1e3
    # A region without wells: fine too (its pressure is arbitrary, but harmless)
    SS, PP = run(ResSim(**kws, wells=pair([0, 0], [.4, 1])))
    assert (SS[-1, right] == 0).all() and np.isfinite(PP).all()
    # Balanced globally, but not per region: caught
    with pytest.raises(AssertionError, match="pressure solve failed"):
        run(ResSim(**kws, wells=pair([0, 0], [1, 1])))
    # ... unless compressible, which needs no pin anywhere
    assert abs(run(ResSim(**kws, ct=.1, wells=pair([0, 0], [1, 1])))[1]).max() < 1e3
    # No active cells at all
    with pytest.raises(AssertionError, match="No active cells"):
        ResSim(Nx=4, Ny=4, active=np.zeros((4, 4)))


def test_well_in_inactive_cell_is_refused():
    active = np.ones((10, 10), bool)
    active[5, 5] = False
    model = ResSim(Nx=10, Ny=10, active=active, wells=pair([.5, .5], [1, 1]))
    with pytest.raises(AssertionError, match="inactive cell"):
        run(model)


def test_plots_mask_the_inactive_cells():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = ResSim(Nx=12, Ny=9, wells=pair([0, 0], [1, 1]))
    active = np.ones(model.shape, bool)
    active[3:6, 2:5] = False
    model.active = active
    SS, PP = run(model)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    cs = model.plt_field(ax1, SS[-1], "oil", finalize=False)
    qm = model.plt_field(ax2, PP[-1], cellwise=True, finalize=False)
    assert qm.get_array().mask.sum() == (~active).sum()
    assert qm.norm.boundaries is not None  # the colours are levelled, like contourf's
    assert cs.get_array() is not None
    # The style dicts still work, `levels` as an array included
    model.plt_field(ax2, SS[-1], "oil", cellwise=True, finalize=False)
    plt.close(fig)
