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

import warnings
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
                   well_xy=[[.5, .5]], well_rates=[[-q]])
    model.well_WI = model.peaceman_WI(model.well_xy, rw)
    SS, PP = model.sim(1e-3, 120, np.zeros(model.Nxy),
                       P0=np.ones(model.Nxy), pbar=False)
    assert SS.max() == 0, "No water is injected, so none should appear."
    dd = PP.mean(axis=1) - PP[:, model.xy2ind(*model.well_xy[0])]
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
    r_e = rw * np.exp(2*np.pi / model.well_WI[0])
    assert np.isclose(r, r_e, rtol=.01)


@pytest.mark.parametrize("N", [16, 32, 64])
def test_bhp_is_grid_independent(N):
    """The cell pressure is a grid artefact; the `bhp` derived from it is not.

    Over `N` = 16 -> 64 the *cell* drawdown grows by some 45% (asserted below),
    whereas the *bottom-hole* drawdown stays put -- at the analytic value, to
    within 0.2%. This is the whole point of having a well model.
    """
    model, PP, dd_cell = depleted(N)
    dd_bhp = PP[-1].mean() - model.actual_bhp[0, -1]
    assert np.isclose(dd_bhp, drawdown_analytic(rw), rtol=3e-3)


def test_cell_pressure_is_not_grid_independent():
    """Lest the above look like a test of nothing."""
    dds = [depleted(N)[2] for N in [16, 32, 64]]
    assert dds[2] / dds[0] > 1.4


def test_bhp_signs():
    """Injectors are *above*, producers *below*, their cell pressure."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=1e-2,
                   well_xy=[[0, 0], [1, 1]], well_rates=[[1], [-1]])
    model.well_WI = model.peaceman_WI(model.well_xy, rw)
    SS, PP = model.sim(.05, 10, np.zeros(model.Nxy), pbar=False)

    for k, (S, P) in enumerate(zip(SS[:-1], PP[1:])):
        for i, sgn in [(0, +1), (1, -1)]:
            ic = model.xy2ind(*model.well_xy[i])
            assert sgn * (model.actual_bhp[i, k] - P[ic]) > 0


def test_bhp_inverts_the_well_model():
    """`bhp` and the well model of `well_WI` are inverses: recover the rates."""
    model, PP, _ = depleted(32)
    Mw, Mo = model.RelPerm(np.zeros(model.Nxy))
    i = model.xy2ind(*model.well_xy[0])
    dp = PP[-1][i] - model.actual_bhp[0, -1]
    assert np.isclose(model.well_WI[0] * (Mw + Mo)[i] * dp, q)


def test_WI_defaults_to_none():
    """Without a well index, `bhp` is `nan` -- and nothing else is affected."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20,
                   well_xy=[[0, 0], [1, 1]], well_rates=[[1], [-1]])
    assert model.well_WI is None
    model.sim(.05, 3, np.zeros(model.Nxy), pbar=False)
    assert np.isnan(model.actual_bhp).all()


def test_WI_normalization():
    """A scalar `WI` broadcasts over the wells; a mis-sized one raises."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20,
                   well_xy=[[0, 0], [1, 1], [1, 0], [0, 1]],
                   well_rates=[[1], [-.3], [-.3], [-.4]])
    model.well_WI = 5.
    assert np.all(model.well_WI == [5., 5., 5., 5.])
    with pytest.raises(ValueError):
        model.well_WI = [1., 2.]


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
# BHP *control* (as opposed to the diagnostic above), ref `ResSim.well_bhp`
# ---------------------------------------------------------------------------

def test_bhp_control_reproduces_rate_control():
    """Rate control and BHP control are two views of the same well model.

    So: run a rate-controlled well, note the `actual_bhp` it implies, then
    prescribe *that* as `well_bhp` -- and recover the original run. To machine
    precision, which is what shows that the coupling is solved for
    simultaneously with the pressure, rather than lagged by a time step.
    """
    def run(**ctrl):
        model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32, ct=.1,
                       well_xy=[[.5, .5]], **ctrl)
        model.well_WI = model.peaceman_WI(model.well_xy, rw)
        _, PP = model.sim(1e-3, 60, np.zeros(model.Nxy),
                          P0=np.ones(model.Nxy), pbar=False)
        return model, PP

    ref, PP = run(well_rates=[[-q]])
    bhp, PP2 = run(well_bhp=ref.actual_bhp)            # NB: `well_rates` unset

    assert np.allclose(PP2, PP, rtol=0, atol=1e-12)
    assert np.allclose(bhp.actual_rates, -q, rtol=0, atol=1e-12)
    # And the diagnostic inverts the control, exactly
    assert np.array_equal(bhp.actual_bhp, ref.actual_bhp)


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
                   well_xy=[[.5, .5]], well_bhp=[[.5]])
    model.well_WI = model.peaceman_WI(model.well_xy, rw)
    dt, nSteps = 2e-3, 150
    _, PP = model.sim(dt, nSteps, np.zeros(model.Nxy),
                      P0=np.ones(model.Nxy), pbar=False)

    rate = -model.actual_rates[0]                    # production is negative
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
                   well_xy=[[0, 0], [1, 1]], well_rates=[[1.], [0]],
                   well_bhp=[[np.nan], [5.]])
    model.well_WI = [np.nan, model.peaceman_WI([[1, 1]], rw)[0]]
    _, PP = model.sim(.02, 10, np.zeros(model.Nxy), pbar=False)

    # Incompressible => no storage => the BHP well must produce all that is injected
    assert np.allclose(model.actual_rates[1], -1.)
    # The level is set by p_bh, not by the (skipped) pin
    assert PP[-1].min() > 5.
    assert np.isclose(model.actual_bhp[1, -1], 5.)


def test_control_modes_may_be_mixed():
    """Across wells, and in time: `nan` entries of `well_bhp` fall back to `well_rates`."""
    nSteps = 8
    schedule = np.where(np.arange(nSteps) < 4, .3, np.nan)  # BHP, then rate
    nans = nSteps*[np.nan]
    model = ResSim(Lx=1, Ly=1, Nx=24, Ny=24, ct=.1,
                   well_xy=[[0, 0], [1, 1], [1, 0]],
                   well_rates=[[.5], [-.2], [-.2]],
                   well_bhp=[nans, schedule, nans])
    model.well_WI = np.append(np.nan, model.peaceman_WI(model.well_xy[1:], rw))
    _, PP = model.sim(.02, nSteps, np.zeros(model.Nxy),
                      P0=np.ones(model.Nxy), pbar=False)

    rates, bhps = model.actual_rates, model.actual_bhp
    assert np.allclose(bhps[1, :4], .3)          # well 1: BHP-controlled...
    assert not np.allclose(rates[1, :4], -.2)    # ...so its rate is not the spec
    assert np.allclose(rates[1, 4:], -.2)        # ...then rate-controlled
    assert np.allclose(rates[2], -.2)            # well 2: always rate-controlled
    assert np.isfinite(bhps[1:]).all()           # but with a WI, so bhp is known
    assert np.isnan(bhps[0]).all()               # the injector has no WI


def test_nan_rate_is_a_placeholder_only_under_bhp_control():
    """`nan` stands in for a BHP-controlled well's (ignored) rate -- and only there.

    Left to a rate-controlled well it would spread through the pressure solve,
    to surface far away as an inscrutable failure; ref `assemble_wells`.
    """
    nSteps = 4
    def run(schedule):
        model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, ct=.1,
                       well_xy=[[0, 0], [1, 1]],
                       well_rates=[[.5], [np.nan]], well_bhp=[nSteps*[np.nan], schedule])
        model.well_WI = [np.nan, model.peaceman_WI([[1, 1]], rw)[0]]
        return model.sim(.02, nSteps, np.zeros(model.Nxy),
                         P0=np.ones(model.Nxy), pbar=False)

    _, PP = run(nSteps*[.5])                      # BHP-controlled throughout: fine
    assert np.isfinite(PP).all()
    with pytest.raises(AssertionError, match="non-finite rate"):
        run([.5, .5, np.nan, np.nan])             # ... but then the `nan` is exposed


def test_bhp_keeps_the_transport_consistent():
    """`realize_bhp` leaves `_Q` equal to the *total* flux, as `storage_rate` needs.

    Cf. `tests/test_compressible.py::test_storage_is_shared`, which asserts the
    same thing for rate control. It is what keeps the saturation step consistent
    with the pressure solution (ref `ResSim.ct`).
    """
    dt = .05
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=1e-2,
                   well_xy=[[0, 0], [1, 1]], well_rates=[[.5], [0]],
                   well_bhp=[[np.nan], [.4]])
    model.well_WI = [np.nan, model.peaceman_WI([[1, 1]], rw)[0]]
    S0 = np.zeros(model.Nxy)
    P0 = np.ones(model.Nxy)

    model.assemble_wells(S0, P0, 0)  # (as `time_stepper` does)
    P, V = model.pressure_step(S0, P0, dt)
    model.realize_bhp(P)

    accum = model.por.ravel() * model.ct * model.h2 / dt
    assert np.allclose(model.storage_rate(V), accum * (P - P0))


def test_bhp_requires_a_well_index():
    """Without a `WI` there is no well model, hence no rate to solve for."""
    model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, ct=.1,
                   well_xy=[[1, 1]], well_bhp=[[.5]])
    with pytest.raises(AssertionError, match="well_WI"):
        model.sim(.02, 2, np.zeros(model.Nxy), P0=np.ones(model.Nxy), pbar=False)


def test_bhp_flow_direction_is_emergent():
    """A BHP well flows whichever way `p_bh` vs. its cell pressure dictates.

    Here its `p_bh` exceeds the initial pressure, so this would-be producer
    instead *injects* -- water, like any inflow -- by design; ref `well_bhp`.
    """
    model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, ct=.1,
                   well_xy=[[.5, .5]], well_bhp=[[2.]])   # above the initial p = 1
    model.well_WI = model.peaceman_WI(model.well_xy, rw)
    SS, PP = model.sim(1e-3, 3, np.zeros(model.Nxy),
                       P0=np.ones(model.Nxy), pbar=False)
    assert np.all(model.actual_rates > 0)   # it flows in...
    assert SS[-1].max() > 0                 # ... which injects water


def test_bhp_flow_reversal_warns():
    """A BHP well that flips direction mid-run does so with a warning.

    Here a rate-controlled injector pressurizes the (closed) field past the
    BHP well's `p_bh`, so the latter starts out injecting (its `p_bh` being
    above the initial pressure) and ends up producing.
    """
    model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, ct=.1,
                   well_xy=[[0, 0], [1, 1]], well_rates=[[.5], [0]],
                   well_bhp=[[np.nan], [1.2]])
    model.well_WI = [np.nan, model.peaceman_WI([[1, 1]], rw)[0]]
    with pytest.warns(UserWarning, match="reversed flow direction"):
        model.sim(.05, 20, np.zeros(model.Nxy),
                  P0=np.ones(model.Nxy), pbar=False)
    rates = model.actual_rates[1]
    assert rates[0] > 0 and rates[-1] < 0


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
    def run(xy, rates, WI=None):
        model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20,
                       well_xy=[*xy, [1, 1]],       # producer appended last
                       well_rates=np.vstack([rates, [[-1.]]]),
                       well_WI=None if WI is None else np.append(WI, np.nan))
        SS, _ = model.sim(.04, 12, np.zeros(model.Nxy), pbar=False)
        return model, SS

    xy, WI, w = ResSim(Lx=1, Ly=1, Nx=20, Ny=20).well_path([[0, 0], [0, .5]], rw)
    horiz, SS_h = run(xy, w[:, None], WI)                    # a 5-cell injector
    point, SS_p = run([[0, 0]], [[1.]])                      # ... vs 1 cell

    # Same total injection (else `time_stepper`'s balance assert would trip)
    assert np.isclose(horiz.actual_rates[:-1].sum(0), 1).all()
    # ... but a different sweep, which is the point of drilling a path
    assert not np.allclose(SS_h[-1], SS_p[-1], atol=1e-3)
    assert (SS_h[-1] > .01).sum() > (SS_p[-1] > .01).sum()   # broader front


def test_path_under_bhp_control_shares_one_pressure():
    """The completions differ in rate, but (absent wellbore effects) not in `p_bh`."""
    xy, WI, _ = ResSim(Lx=1, Ly=1, Nx=20, Ny=20).well_path([[0, 0], [0, .5]], rw)
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=.1,
                   well_xy=[*xy, [1, 1]],           # producer appended last
                   well_rates=np.vstack([np.zeros((len(xy), 1)), [[-.5]]]),
                   well_WI=np.append(WI, np.nan),
                   well_bhp=np.vstack([np.full((len(xy), 1), 3.), [[np.nan]]]))
    model.sim(.02, 6, np.zeros(model.Nxy), P0=np.ones(model.Nxy), pbar=False)

    rates, bhps = model.actual_rates[:-1], model.actual_bhp[:-1]
    assert np.allclose(bhps, 3.)                  # one wellbore, one pressure
    assert rates.min() > 0                        # all completions inject
    assert rates[:, -1].std() > 1e-3              # but not equally


# ---------------------------------------------------------------------------
# Feedback control, i.e. state-dependent controls, ref `ResSim.well_controls`
# ---------------------------------------------------------------------------

class Shutter(ResSim):
    """Shuts all the wells upon water arrival at the producer."""

    def well_controls(self, S, P, k):
        ctrl = super().well_controls(S, P, k)
        if S is not None and S[self.xy2ind(*self.well_xy[1])] > .5:
            ctrl["rates"][:] = 0
        return ctrl


def waterflood(cls=ResSim, **kwargs):
    """A quarter-five-spot, run past breakthrough. Cf. `examples/quarter_five_spot.py`."""
    model = cls(Lx=1, Ly=1, Nx=16, Ny=16,
                well_xy=[[0, 0], [1, 1]], well_rates=[[1.], [-1.]], **kwargs)
    SS, _ = model.sim(.05, 20, model.swc*np.ones(model.Nxy), pbar=False)
    return model, SS


def test_rate_feedback_shuts_the_well():
    """The hook overrides the schedule, and the shut-in shows up in `actual_rates`."""
    _, SS0 = waterflood()
    model, SS = waterflood(Shutter)

    kShut = int((model.actual_rates[1] == 0).argmax())
    assert 0 < kShut < 20                            # it did happen, mid-run
    assert np.allclose(model.actual_rates[1, :kShut], -1.)   # ... and only then
    assert np.allclose(model.actual_rates[1, kShut:], 0.)
    assert np.allclose(model.actual_rates[0], -model.actual_rates[1])
    # Once shut, nothing more moves; whereas the reference floods on
    assert np.allclose(SS[-1], SS[kShut])
    assert SS0[-1].sum() > SS[-1].sum()


def test_rate_feedback_does_not_modify_the_spec():
    """The hook gets copies, so an in-place override cannot corrupt the schedule."""
    model, _ = waterflood(Shutter)
    assert np.all(model.well_rates == [[1.], [-1.]])


def test_rate_feedback_must_balance_when_incompressible():
    """Shutting only one side of an incompressible model is caught, not leaked."""
    class HalfShutter(Shutter):
        def well_controls(self, S, P, k):
            ctrl = super().well_controls(S, P, k)
            ctrl["rates"][0] = 1.   # undo the injector's share of the shut-in
            return ctrl

    with pytest.raises(AssertionError, match="sum to 0"):
        waterflood(HalfShutter)


def test_rate_feedback_may_act_alone_when_compressible():
    """With `ct > 0` storage absorbs the imbalance, so the producer needs no partner."""
    class Producer(ResSim):
        def well_controls(self, S, P, k):
            ctrl = super().well_controls(S, P, k)
            if S is not None and S[self.xy2ind(*self.well_xy[1])] > .5:
                ctrl["rates"][1] = 0    # NB: injector left flowing
            return ctrl

    model, _ = waterflood(Producer, ct=1e-2)
    kShut = int((model.actual_rates[1] == 0).argmax())
    assert 0 < kShut < 20
    assert np.allclose(model.actual_rates[0], 1.)


def test_well_controls_reads_the_specs():
    """The default hook is a pure lookup, and reports both kinds of control."""
    model = ResSim(Lx=1, Ly=1, Nx=8, Ny=8,
                   well_xy=[[0, 0], [1, 1], [1, 0]],
                   well_bhp=[[.5], [np.nan], [np.nan]])
    ctrl = model.well_controls(None, None, 0)

    assert set(ctrl) == {"rates", "bhp"}
    assert np.array_equal(ctrl["rates"], [0., 0., 0.])   # unset ⇒ 0
    assert np.array_equal(ctrl["bhp"], [.5, np.nan, np.nan], equal_nan=True)

    model.well_rates = [[1.], [-.5], [-.5]]
    ctrl = model.well_controls(None, None, 0)
    assert np.array_equal(ctrl["rates"], [1., -.5, -.5])


class Limited(ResSim):
    """Rate control with a BHP limit: the (approximate) mode switch of `well_bhp`."""

    p_min = .5

    def well_controls(self, S, P, k):
        ctrl = super().well_controls(S, P, k)
        if P is None:
            return ctrl
        p_bh = self.bhp(S, P, ctrl["rates"])          # what the rate would require
        ctrl["bhp"] = np.where(p_bh < self.p_min, self.p_min, np.nan)
        return ctrl


def deplete(cls=ResSim, dt=2e-3, nSteps=150, **kwargs):
    """Deplete a closed square at a fixed rate. Cf. `examples/well_control.py`."""
    model = cls(Lx=1, Ly=1, Nx=32, Ny=32, ct=.1,
                well_xy=[[.5, .5]], well_rates=[[-q]], **kwargs)
    model.well_WI = model.peaceman_WI(model.well_xy, rw)
    model.sim(dt, nSteps, np.zeros(model.Nxy), P0=np.ones(model.Nxy), pbar=False)
    return -model.actual_rates[0], model.actual_bhp[0]   # production as positive


def test_well_controls_may_switch_the_mode():
    """A well may thus be *rate*-controlled until its BHP limit binds."""
    rate0, bhp0 = deplete()             # for reference: no limit
    rate, bhp = deplete(Limited)
    p_min = Limited.p_min

    assert bhp0.min() < p_min           # the unlimited run does breach the limit
    kSwitch = int((bhp0 < p_min).argmax())
    assert 0 < kSwitch < len(bhp0)
    # Up to and including the switch step -- which the lag leaves rate-controlled
    # -- the run is identical to the unlimited one, and breaches the limit as it did
    assert np.allclose(rate[:kSwitch + 1], q)
    assert np.allclose(bhp[:kSwitch + 1], bhp0[:kSwitch + 1])
    # Thereafter: BHP-controlled at the limit, and the rate must give way
    assert np.allclose(bhp[kSwitch + 1:], p_min)
    assert np.all(rate[kSwitch + 1:] < q)
    assert rate[-1] < q/2


def test_well_controls_switch_lags_by_a_step():
    """Being decided from the previous pressure, the limit is breached briefly.

    But only for the one step, and by less the smaller `dt` is.
    """
    def overshoot(dt, nSteps):
        _, bhp = deplete(Limited, dt, nSteps)
        return Limited.p_min - bhp.min()

    coarse = overshoot(4e-3, 75)
    fine = overshoot(1e-3, 300)
    assert 0 < fine < coarse / 3


# ---------------------------------------------------------------------------
# The wells on a plot, i.e. their sign, ref `TPFA_ResSim.plotting._well_signs`
# ---------------------------------------------------------------------------

def test_well_signs_read_the_spec():
    """Injector `+1`, producer `-1`, undecided `0` -- by the sign of the rates."""
    model = ResSim(Lx=1, Ly=1, Nx=8, Ny=8,
                   well_xy=[[0, 0], [1, 1], [1, 0]],
                   well_rates=[[1.], [-.5], [0]])
    assert list(model._well_signs()) == [+1, -1, 0]


def test_well_signs_skip_nan_placeholders():
    """A BHP well's `nan` rate leaves it undecided -- rather than cast to garbage.

    NB: `np.sign(nan).astype(int)` is not merely `nan`-valued, but *undefined*
    (and warns), which is why the sum ignores the `nan`s.
    """
    model = ResSim(Lx=1, Ly=1, Nx=8, Ny=8,
                   well_xy=[[0, 0], [1, 1]],
                   well_rates=[[np.nan], [-1.]], well_bhp=[[3.], [np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert list(model._well_signs()) == [0, -1]


def test_well_signs_fall_back_on_the_realized_rates():
    """What a well the spec leaves undecided (pure BHP control) actually did."""
    model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, ct=.1,
                   well_xy=[[0, 0], [1, 1]],
                   well_rates=[[0], [-.5]], well_bhp=[[3.], [np.nan]])
    model.well_WI = [model.peaceman_WI([[0, 0]], rw)[0], np.nan]
    assert list(model._well_signs()) == [0, -1]     # before the sim: undecided
    model.sim(.02, 4, np.zeros(model.Nxy), P0=np.ones(model.Nxy), pbar=False)
    assert list(model._well_signs()) == [+1, -1]    # after it: it injected
    assert np.all(model.actual_rates[0] > 0)


def test_well_markers_are_numbered_per_sign():
    """As `plt_production` numbers them: per sign, not as in the unified `well_xy`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = ResSim(Lx=1, Ly=1, Nx=8, Ny=8,
                   well_xy=[[0, 0], [1, 1], [1, 0]],
                   well_rates=[[1.], [-.5], [-.5]])
    _, ax = plt.subplots()
    try:
        model.plt_field(ax, np.zeros(model.Nxy), finalize=False)
        labels = sorted(t.get_text() for t in ax.texts)
    finally:
        plt.close("all")
    # I.e. producers 0 and 1, and injector 0 -- not wells 0, 1 and 2
    assert labels == ["0", "0", "1"]
