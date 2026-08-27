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
                       p0=np.ones(model.Nxy), pbar=False)
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
