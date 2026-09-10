"""The fluid (`ResSim.fluid`): its Corey relative permeabilities and their
derivatives, the fractional flow, the `fluid` normalization, and the CFL bound
that follows the curves."""

import copy
import pickle

import numpy as np
import pytest

from minires import Fluid, ResSim
from minires.grid import Fluxes

# The Egg model's `SWOF` table (examples/egg.py): Sw, krw, kro. Corey 3/4 with
# end-points 0.6/0.8 on [0.2, 0.85]; the table's last row (0.9) is beyond 1 - sor,
# where the oil is immobile and its krw (0.749) is the polynomial run on.
EGG_SWOF = np.array([
    [0.10, 0.0000e+00, 8.0000e-01], [0.20, 0.0000e+00, 8.0000e-01],
    [0.25, 2.7310e-04, 5.8082e-01], [0.30, 2.1848e-03, 4.1010e-01],
    [0.35, 7.3737e-03, 2.8010e-01], [0.40, 1.7478e-02, 1.8378e-01],
    [0.45, 3.4138e-02, 1.1473e-01], [0.50, 5.8990e-02, 6.7253e-02],
    [0.55, 9.3673e-02, 3.6301e-02], [0.60, 1.3983e-01, 1.7506e-02],
    [0.65, 1.9909e-01, 7.1706e-03], [0.70, 2.7310e-01, 2.2688e-03],
    [0.75, 3.6350e-01, 4.4820e-04], [0.80, 4.7192e-01, 2.8000e-05],
    [0.85, 6.0000e-01, 0.0000e+00], [0.90, 7.4939e-01, 0.0000e+00],
])
EGG: dict = dict(swc=0.2, sor=0.15, nw=3, no=4, krw0=0.6, kro0=0.8)


def model(fluid=None):
    return ResSim(Lx=1, Ly=1, Nx=2, Ny=2, fluid=fluid)


def test_fluid_normalization():
    """`fluid` takes a dict, `None` (the defaults) or an instance; the fields
    stay writable; and instances are independent (as an ensemble needs)."""
    a, b = model(), model()
    assert isinstance(a.fluid, Fluid) and a.fluid == Fluid() and a.fluid is not b.fluid
    a.fluid.swc = .3
    assert b.fluid.swc == 0
    m = model(EGG)
    assert m.fluid == Fluid(**EGG)
    m.fluid = Fluid(vo=5)
    assert m.fluid.vo == 5
    m.fluid = None
    assert m.fluid == Fluid()
    # The nested repr, and that copies survive (`_pLU` aside, nothing exotic)
    assert "fluid: Fluid(" in repr(m)
    assert pickle.loads(pickle.dumps(m)).fluid == copy.deepcopy(m).fluid == Fluid()


def test_custom_curves():
    """Curves of another shape are a subclass overriding `RelPerm`/`dRelPerm`:
    here straight lines. `fractional_flow` and the CFL bound follow."""
    class Linear(Fluid):
        def RelPerm(self, s):
            S = np.clip(self.rescale_sat(s), 0, 1)
            return S / self.vw, (1 - S) / self.vo

        def dRelPerm(self, s):
            S = self.rescale_sat(s)
            inside = ((0 <= S) & (S <= 1)) / (1 - self.swc - self.sor)
            return inside / self.vw, -inside / self.vo

    m = model(Linear(vo=2., swc=.2, sor=.1))
    s = np.array([.1, .55, .95])
    Mw, Mo = m.fluid.RelPerm(s)
    assert np.allclose(Mw, [0, .5, 1]) and np.allclose(Mo, [.5, .25, 0])
    fw, dfw = m.fluid.fractional_flow(s), m.fluid.dfractional_flow(s)
    assert np.allclose(fw, [0, 2/3, 1])  # f_w = 2S/(1+S)
    assert np.allclose(dfw, [0, 2/1.5**2/.7, 0])
    # The slope peaks at S = 0, a sample point, so the bound is exact here
    assert np.isclose(dfw_max(m), 2/.7, rtol=1e-12)


def test_defaults_are_quadratic():
    """The defaults reproduce the reference paper's curves (Listing 6)."""
    m = model(dict(swc=.2, sor=.1, vw=1., vo=3.))
    s = np.linspace(.2, .9, 15)
    S = (s - .2) / .7
    Mw, Mo = m.fluid.RelPerm(s)
    assert np.allclose(Mw, S**2) and np.allclose(Mo, (1 - S)**2 / 3)


def test_egg_swof_table():
    """The Corey parameters reproduce the Egg deck's table to its 4-5 digits."""
    m = model(EGG)
    Mw, Mo = m.fluid.RelPerm(EGG_SWOF[:-1, 0])
    assert np.allclose(Mw, EGG_SWOF[:-1, 1], atol=1e-4)
    assert np.allclose(Mo, EGG_SWOF[:-1, 2], atol=1e-4)


def test_clipping():
    """Below/above the residuals, a phase is immobile -- not negative or > 1."""
    m = model(EGG)
    Mw, Mo = m.fluid.RelPerm(np.array([-.1, 0., .1, .2]))
    assert (Mw == 0).all() and (Mo == .8).all()
    Mw, Mo = m.fluid.RelPerm(np.array([.85, .9, 1., 1.1]))
    assert np.allclose(Mo, 0, atol=1e-15) and np.allclose(Mw, .6)


@pytest.mark.parametrize("fluid", [dict(), dict(swc=.2, sor=.1, vo=5.), EGG])
def test_derivative(fluid):
    """`dRelPerm` is the derivative of `RelPerm`, and `dfractional_flow` that of
    `fractional_flow` (central differences, interior)."""
    f = model(fluid).fluid
    lo, hi = f.swc, 1 - f.sor
    s = np.linspace(lo, hi, 41)[1:-1]
    h = 1e-6
    dMw, dMo = f.dRelPerm(s)
    Mw1, Mo1 = f.RelPerm(s + h)
    Mw0, Mo0 = f.RelPerm(s - h)
    assert np.allclose(dMw, (Mw1 - Mw0) / (2*h), rtol=1e-6, atol=1e-8)
    assert np.allclose(dMo, (Mo1 - Mo0) / (2*h), rtol=1e-6, atol=1e-8)
    fw, dfw = f.fractional_flow(s), f.dfractional_flow(s)
    assert np.allclose(fw, Mw1 / (Mw1 + Mo1), atol=1e-5)
    fw1, fw0 = f.fractional_flow(s + h), f.fractional_flow(s - h)
    assert np.allclose(dfw, (fw1 - fw0) / (2*h), rtol=1e-6, atol=1e-8)
    # Outside: clipped, hence flat; at the ends: one-sided (the reference code's values)
    assert f.dRelPerm(np.array([lo - .05, hi + .05]))[0].tolist() == [0, 0]
    dMw, dMo = f.dRelPerm(np.array([lo, hi]))
    assert np.isclose(dMo[0], -f.kro0 * f.no / (hi - lo) / f.vo)
    assert np.isclose(dMw[1], f.krw0 * f.nw / (hi - lo) / f.vw)


def dfw_max(m):
    """The maximal fractional-flow slope, as `estimate_1CFL` bounds it: its
    estimate for a unit influx per unit pore volume, less its safety factor."""
    V = Fluxes(np.zeros((m.Nx + 1, m.Ny)), np.zeros((m.Nx, m.Ny + 1)))
    return m.estimate_1CFL(np.ones(m.Nxy), V, np.ones(m.Nxy)) / 1.5


def test_dfw_max():
    """The maximal fractional-flow slope: analytic for the default, and never
    exceeded (beyond the sampling's undershoot) by the curve for the others."""
    m = model(dict(swc=.1, sor=.2))
    assert np.isclose(dfw_max(m), 2 / (1 - .1 - .2), rtol=1e-5)
    for fluid in [dict(vo=5.), dict(swc=.2, sor=.1, vo=5.), EGG]:
        m = model(fluid)
        s = np.linspace(m.fluid.swc, 1 - m.fluid.sor, 100001)
        fw = m.fluid.fractional_flow(s)
        assert dfw_max(m) >= np.gradient(fw, s).max() * (1 - 1e-5)
    assert np.isclose(dfw_max(model(dict(**EGG, vo=5.))), 5.65, atol=.01)  # ref examples/egg.py


def test_cfl_follows_the_curves():
    """Steeper curves demand more explicit sub-steps: the run stays monotone."""
    m = ResSim(Lx=1, Ly=1, Nx=10, Ny=10, fluid=dict(**EGG, vo=5.),
               wells=[dict(xy=[0, 0], rate=1), dict(xy=[1, 1], rate=-1)])
    SS, _ = m.sim(.1, 5, np.full(m.Nxy, .1), pbar=False)
    assert (SS >= .1 - 1e-12).all() and (SS <= .9 + 1e-12).all()
