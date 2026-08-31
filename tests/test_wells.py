"""Tests of the well model, i.e. of the well index (`ResSim.peaceman_WI`).

Like `test_compressible.py` (and unlike the examples) there is no Matlab
reference to compare with: the Matlab codes have no well model at all, their
wells being plain source terms. Instead we verify against *analytic* well
testing theory -- which happens to be possible here, and rather sharply so.

The setting is a single producer at the centre of a closed (no-flow) square,
run until the flow is *boundary-dominated* (ref `examples/depletion.py`), where
the drawdown from the average pressure is given by the pseudo-steady-state
solution with Dietz shape factor $ C_A = 30.8828 $:
$$ \\bar{p} - p(r) = \\frac{q}{2 π k λ_t} \\,
                     \\frac{1}{2} \\ln \\frac{4 A}{e^γ \\, C_A \\, r^2} \\,. $$
"""

from functools import cache

import numpy as np
import pytest

from TPFA_ResSim import ResSim

C_A = 30.8828       # Dietz shape factor, well at the centre of a square
e_gamma = 1.781072  # exp(Euler--Mascheroni)
q = .25             # production rate (the only well, so also the voidage)
A = 1.              # area of the (unit) domain
rw = 1e-3           # well radius


def drawdown_analytic(r):
    """Pseudo-steady-state $ \\bar{p} - p(r) $, per the docstring above ($k = λ_t = 1$)."""
    return q / (2*np.pi) * .5*np.log(4*A / (e_gamma * C_A * r**2))


@cache
def depleted(N, ct=.1):
    """Deplete a closed square via a central producer; return it, past its transient.

    Single-phase (there is no water anywhere), so $ λ_t = M_o = 1/v_o = 1 $.
    Cf. `examples/depletion.py`.
    """
    model = ResSim(Lx=1, Ly=1, Nx=N, Ny=N, ct=ct,
                   inj_xy=[[0, 0]]  , inj_rates=[[0]],
                   prd_xy=[[.5, .5]], prd_rates=[[q]])
    model.prd_WI = model.peaceman_WI(model.prd_xy, rw)
    SS, PP = model.sim(1e-3, 120, np.zeros(model.Nxy),
                       P0=np.ones(model.Nxy), pbar=False)
    assert SS.max() == 0, "No water is injected, so none should appear."
    dd = PP.mean(axis=1) - PP[:, model.xy2ind(*model.prd_xy[0])]
    assert np.isclose(dd[-1], dd[-2], rtol=1e-3), "Not yet boundary-dominated."
    return model, PP, dd[-1]


def test_equivalent_radius_formula():
    """$ r_e = .28 \\sqrt{2} h / 2 ≈ .198 h $ on an isotropic, square grid."""
    model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32)
    r_e = rw * np.exp(2*np.pi / model.peaceman_WI([[.5, .5]], rw)[0])
    assert np.isclose(r_e / model.hx, .198, atol=1e-3)


@pytest.mark.parametrize("N", [16, 32, 64])
def test_equivalent_radius_is_realized(N):
    """The *cell* pressure really does sit at Peaceman's $ r_e $.

    I.e. invert `drawdown_analytic` for the radius, and compare. Note that this
    is a property of the 5-point stencil that `TPFA` assembles, rather than of
    the well model -- which is precisely why the well model may rely on it.
    """
    model, _, dd_cell = depleted(N)
    r = np.sqrt(4*A / (e_gamma * C_A * np.exp(2 * dd_cell / (q / (2*np.pi)))))
    r_e = rw * np.exp(2*np.pi / model.prd_WI[0])
    assert np.isclose(r, r_e, rtol=.01)


@pytest.mark.parametrize("N", [16, 32, 64])
def test_bhp_is_grid_independent(N):
    """The cell pressure is a grid artefact; the `bhp` derived from it is not.

    Over `N` = 16 -> 64 the *cell* drawdown grows by some 45% (asserted below),
    whereas the *bottom-hole* drawdown stays put -- at the analytic value, to
    within 0.2%. This is the whole point of having a well model.
    """
    model, PP, dd_cell = depleted(N)
    dd_bhp = PP[-1].mean() - model.actual_bhp["prd"][0, -1]
    assert np.isclose(dd_bhp, drawdown_analytic(rw), rtol=3e-3)


def test_cell_pressure_is_not_grid_independent():
    """Lest the above look like a test of nothing."""
    dds = [depleted(N)[2] for N in [16, 32, 64]]
    assert dds[2] / dds[0] > 1.4


def test_bhp_signs():
    """Injectors are *above*, producers *below*, their cell pressure."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=1e-2,
                   inj_xy=[[0, 0]], inj_rates=[[1]],
                   prd_xy=[[1, 1]], prd_rates=[[1]])
    for kind in ["inj", "prd"]:
        setattr(model, f"{kind}_WI", model.peaceman_WI(getattr(model, f"{kind}_xy"), rw))
    SS, PP = model.sim(.05, 10, np.zeros(model.Nxy), pbar=False)

    for k, (S, P) in enumerate(zip(SS[:-1], PP[1:])):
        for kind, sgn in [("inj", +1), ("prd", -1)]:
            i = model.xy2ind(*getattr(model, f"{kind}_xy")[0])
            assert sgn * (model.actual_bhp[kind][0, k] - P[i]) > 0


def test_bhp_inverts_the_well_model():
    """`bhp` and the well model of `inj_WI` are inverses: recover the rates."""
    model, PP, _ = depleted(32)
    Mw, Mo = model.RelPerm(np.zeros(model.Nxy))
    i = model.xy2ind(*model.prd_xy[0])
    dp = PP[-1][i] - model.actual_bhp["prd"][0, -1]
    assert np.isclose(model.prd_WI[0] * (Mw + Mo)[i] * dp, q)


def test_WI_defaults_to_none():
    """Without a well index, `bhp` is `nan` -- and nothing else is affected."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20,
                   inj_xy=[[0, 0]], inj_rates=[[1]],
                   prd_xy=[[1, 1]], prd_rates=[[1]])
    assert model.inj_WI is None and model.prd_WI is None
    model.sim(.05, 3, np.zeros(model.Nxy), pbar=False)
    assert np.isnan(model.actual_bhp["prd"]).all()
    assert np.isnan(model.actual_bhp["inj"]).all()


def test_WI_normalization():
    """A scalar `WI` broadcasts over the wells; a mis-sized one raises."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20,
                   inj_xy=[[0, 0]], inj_rates=[[1]],
                   prd_xy=[[1, 1], [1, 0], [0, 1]], prd_rates=[[.3], [.3], [.4]])
    model.prd_WI = 5.
    assert np.all(model.prd_WI == [5., 5., 5.])
    with pytest.raises(ValueError):
        model.prd_WI = [1., 2.]


def test_WI_anisotropic():
    """With $ k_x = k_y $ the anisotropic formula reduces to the isotropic one."""
    iso = ResSim(Lx=1, Ly=2, Nx=16, Ny=8, K=3.)
    ani = ResSim(Lx=1, Ly=2, Nx=16, Ny=8,
                 K=np.stack([3*np.ones((16, 8)), 12*np.ones((16, 8))]))
    xy = [[.5, 1.]]
    assert np.isclose(iso.peaceman_WI(xy, rw)[0],
                      2*np.pi*3 / np.log(.28*np.sqrt(iso.hx**2 + iso.hy**2)/2 / rw))
    # sqrt(kx ky) = 6 > 3: the anisotropic well is the more productive
    assert ani.peaceman_WI(xy, rw)[0] > iso.peaceman_WI(xy, rw)[0]


def test_skin():
    """Positive skin (near-well damage) reduces the well index."""
    model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32)
    WI = [model.peaceman_WI([[.5, .5]], rw, skin=s)[0] for s in [-1, 0, 3]]
    assert WI[0] > WI[1] > WI[2] > 0


# ---------------------------------------------------------------------------
# BHP *control* (as opposed to the diagnostic above), ref `ResSim.inj_bhp`
# ---------------------------------------------------------------------------

def test_bhp_control_reproduces_rate_control():
    """Rate control and BHP control are two views of the same well model.

    So: run a rate-controlled well, note the `actual_bhp` it implies, then
    prescribe *that* as `prd_bhp` -- and recover the original run. To machine
    precision, which is what shows that the coupling is solved for
    simultaneously with the pressure, rather than lagged by a time step.
    """
    def run(**ctrl):
        model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32, ct=.1,
                       inj_xy=[[0, 0]], inj_rates=[[0]], prd_xy=[[.5, .5]], **ctrl)
        model.prd_WI = model.peaceman_WI(model.prd_xy, rw)
        _, PP = model.sim(1e-3, 60, np.zeros(model.Nxy),
                          P0=np.ones(model.Nxy), pbar=False)
        return model, PP

    ref, PP = run(prd_rates=[[q]])
    bhp, PP2 = run(prd_bhp=ref.actual_bhp["prd"])      # NB: `prd_rates` unset

    assert np.allclose(PP2, PP, rtol=0, atol=1e-12)
    assert np.allclose(bhp.actual_rates["prd"], q, rtol=0, atol=1e-12)
    # And the diagnostic inverts the control, exactly
    assert np.array_equal(bhp.actual_bhp["prd"], ref.actual_bhp["prd"])


@pytest.mark.parametrize("N", [32, 64])
def test_bhp_depletion_declines_exponentially(N):
    """At constant $ p_\\mathrm{bh} $, material balance gives $ q ∝ e^{-t/τ} $.

    Combining $ c_t V_p \\, d\\bar{p}/dt = -q $ with $ q = J (\\bar{p} - p_bh) $
    yields $ τ = c_t V_p / J $, where the productivity index $ J $ is
    `drawdown_analytic` inverted. Note that this tests the *rate* that the model
    solves for -- and that, like the drawdown, it is grid-independent.
    """
    ct = .1
    J = q / drawdown_analytic(rw)
    model = ResSim(Lx=1, Ly=1, Nx=N, Ny=N, ct=ct,
                   inj_xy=[[0, 0]], inj_rates=[[0]],
                   prd_xy=[[.5, .5]], prd_bhp=[[.5]])
    model.prd_WI = model.peaceman_WI(model.prd_xy, rw)
    dt, nSteps = 2e-3, 150
    _, PP = model.sim(dt, nSteps, np.zeros(model.Nxy),
                      P0=np.ones(model.Nxy), pbar=False)

    rate = model.actual_rates["prd"][0]
    tt = dt * np.arange(1, nSteps + 1)
    late = tt > .1                                   # past the transient
    tau = -1 / np.polyfit(tt[late], np.log(rate[late]), 1)[0]
    assert np.isclose(tau, ct / J, rtol=.03)         # $ V_p = 1 $
    # The instantaneous rate obeys the same J (once boundary-dominated)
    pbar = PP[1:].mean(axis=1)
    assert np.allclose(rate[late], J * (pbar - .5)[late], rtol=.02)


def test_bhp_anchors_the_incompressible_pressure():
    """With `ct == 0`, a BHP well removes *both* of the pure-Neumann caveats.

    I.e. the rates need no longer balance (the well finds its own), and the
    pressure is no longer defined merely up to a constant -- so `TPFA` must skip
    the pin that it would otherwise apply at the SW corner. That pin is exactly
    what the balance assertion below would detect: it acts as a spurious well,
    draining to `p = 0`, so production would fall short of injection.
    """
    model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32,   # NB: ct = 0
                   inj_xy=[[0, 0]], inj_rates=[[1.]],
                   prd_xy=[[1, 1]], prd_bhp=[[5.]])
    model.prd_WI = model.peaceman_WI(model.prd_xy, rw)
    _, PP = model.sim(.02, 10, np.zeros(model.Nxy), pbar=False)

    # Incompressible => no storage => the BHP well must produce all that is injected
    assert np.allclose(model.actual_rates["prd"], 1.)
    # The level is set by p_bh, not by the (skipped) pin
    assert PP[-1].min() > 5.
    assert np.isclose(model.actual_bhp["prd"][0, -1], 5.)


def test_control_modes_may_be_mixed():
    """Across wells, and in time: `nan` entries of `*_bhp` fall back to `*_rates`."""
    nSteps = 8
    schedule = np.where(np.arange(nSteps) < 4, .3, np.nan)  # BHP, then rate
    model = ResSim(Lx=1, Ly=1, Nx=24, Ny=24, ct=.1,
                   inj_xy=[[0, 0]], inj_rates=[[.5]],
                   prd_xy=[[1, 1], [1, 0]],
                   prd_rates=[[.2], [.2]], prd_bhp=[schedule, nSteps*[np.nan]])
    model.prd_WI = model.peaceman_WI(model.prd_xy, rw)
    _, PP = model.sim(.02, nSteps, np.zeros(model.Nxy),
                      P0=np.ones(model.Nxy), pbar=False)

    rates, bhps = model.actual_rates["prd"], model.actual_bhp["prd"]
    assert np.allclose(bhps[0, :4], .3)          # well 0: BHP-controlled...
    assert not np.allclose(rates[0, :4], .2)     # ...so its rate is not the spec
    assert np.allclose(rates[0, 4:], .2)         # ...then rate-controlled
    assert np.allclose(rates[1], .2)             # well 1: always rate-controlled
    assert np.isfinite(bhps).all()               # but with a WI, so bhp is known


def test_bhp_keeps_the_transport_consistent():
    """`_realize_bhp` leaves `_Q` equal to the *total* flux, as `storage_rate` needs.

    Cf. `tests/test_compressible.py::test_storage_is_shared`, which asserts the
    same thing for rate control. It is what keeps the saturation step consistent
    with the pressure solution (ref `ResSim.ct`).
    """
    dt = .05
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=1e-2,
                   inj_xy=[[0, 0]], inj_rates=[[.5]],
                   prd_xy=[[1, 1]], prd_bhp=[[.4]])
    model.prd_WI = model.peaceman_WI(model.prd_xy, rw)
    S0 = np.zeros(model.Nxy)
    P0 = np.ones(model.Nxy)

    model._set_Q(S0, 0)                       # (as `time_stepper` does)
    P, V = model.pressure_step(S0, P0, dt)
    model._realize_bhp(P, 0)

    accum = model.por.ravel() * model.ct * model.h2 / dt
    assert np.allclose(model.storage_rate(V), accum * (P - P0))


def test_bhp_requires_a_well_index():
    """Without a `WI` there is no well model, hence no rate to solve for."""
    model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, ct=.1,
                   inj_xy=[[0, 0]], inj_rates=[[0]],
                   prd_xy=[[1, 1]], prd_bhp=[[.5]])
    with pytest.raises(AssertionError, match="requires `prd_WI`"):
        model.sim(.02, 2, np.zeros(model.Nxy), P0=np.ones(model.Nxy), pbar=False)


def test_backflow_is_rejected():
    """A producer whose `p_bh` exceeds its cell pressure would inject. Ref `inj_bhp`."""
    model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, ct=.1,
                   inj_xy=[[0, 0]], inj_rates=[[0]],
                   prd_xy=[[.5, .5]], prd_bhp=[[2.]])   # above the initial p = 1
    model.prd_WI = model.peaceman_WI(model.prd_xy, rw)
    with pytest.raises(AssertionError, match="flow backwards"):
        model.sim(1e-3, 3, np.zeros(model.Nxy), P0=np.ones(model.Nxy), pbar=False)


# ---------------------------------------------------------------------------
# Well *paths*, i.e. multi-cell completions, ref `ResSim.well_path`
# ---------------------------------------------------------------------------

def path_lengths(model, vertices):
    """The traversed length per cell, recovered from `well_path`'s `WI`."""
    xy, WI, _ = model.well_path(vertices, rw)
    return xy, WI / model.peaceman_WI(xy, rw) * np.sqrt(model.h2)


@pytest.mark.parametrize("vertices", [
    [[.05, .05], [.95, .05]],              # straight, axis-aligned
    [[.05, .05], [.95, .95]],              # diagonal
    [[.05, .05], [.05, .95], [.95, .95]],  # L-shaped (2 segments)
    [[.13, .07], [.62, .41]],              # neither aligned nor through centres
])
def test_path_conserves_length(vertices):
    """The per-cell traversals must partition the polyline."""
    model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10)
    _, lengths = path_lengths(model, vertices)
    V = np.asarray(vertices)
    expected = np.hypot(*np.diff(V, axis=0).T).sum()
    assert np.isclose(lengths.sum(), expected)


def test_path_geometry():
    """A path along a row of cells: the cells it enters, and its share of each."""
    model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10)
    xy, WI, w = model.well_path([[.05, .05], [.45, .05]], rw)

    assert np.allclose(xy[:, 0], [.05, .15, .25, .35, .45])   # 5 cells, in x
    assert np.allclose(xy[:, 1], .05)                          # all in one row
    assert np.allclose(w, [.125, .25, .25, .25, .125])         # ends are halved
    assert np.isclose(w.sum(), 1)
    # A full crossing scores the cell's undiminished well index
    assert np.allclose(WI[1:4], model.peaceman_WI(xy[1:4], rw))


def test_path_revisiting_a_cell_accumulates():
    """A path that doubles back reports each cell once, with the total length."""
    model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10)
    xy, lengths = path_lengths(model, [[.05, .05], [.35, .05], [.15, .05]])
    assert len(xy) == 4                             # cells 0..3, not 4+3
    assert np.isclose(lengths.sum(), .3 + .2)


def test_path_under_rate_control_matches_a_point_well_in_total():
    """Superimposition: the completions act as one well, of the given total rate."""
    def run(**wells):
        model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20,
                       prd_xy=[[1, 1]], prd_rates=[[1.]], **wells)
        SS, _ = model.sim(.04, 12, np.zeros(model.Nxy), pbar=False)
        return model, SS

    xy, _, w = ResSim(Lx=1, Ly=1, Nx=20, Ny=20).well_path([[0, 0], [0, .5]], rw)
    horiz, SS_h = run(inj_xy=xy, inj_rates=w[:, None])       # a 5-cell injector
    point, SS_p = run(inj_xy=[[0, 0]], inj_rates=[[1.]])     # ... vs 1 cell

    # Same total injection (else `time_stepper`'s balance assert would trip)
    assert np.isclose(horiz.actual_rates["inj"].sum(0), 1).all()
    # ... but a different sweep, which is the point of drilling a path
    assert not np.allclose(SS_h[-1], SS_p[-1], atol=1e-3)
    assert (SS_h[-1] > .01).sum() > (SS_p[-1] > .01).sum()   # broader front


def test_path_under_bhp_control_shares_one_pressure():
    """The completions differ in rate, but (absent wellbore effects) not in `p_bh`."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=.1,
                   prd_xy=[[1, 1]], prd_rates=[[.5]])
    xy, WI, _ = model.well_path([[0, 0], [0, .5]], rw)
    model.inj_xy, model.inj_WI = xy, WI
    model.inj_bhp = np.full((len(xy), 1), 3.)
    model.sim(.02, 6, np.zeros(model.Nxy), P0=np.ones(model.Nxy), pbar=False)

    rates, bhps = model.actual_rates["inj"], model.actual_bhp["inj"]
    assert np.allclose(bhps, 3.)                  # one wellbore, one pressure
    assert rates.min() > 0                        # all completions inject
    assert rates[:, -1].std() > 1e-3              # but not equally
