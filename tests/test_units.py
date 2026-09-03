"""Tests of `ResSim.cdarcy`, i.e. of the units.

The equations here contain no conversion factors, so any *coherent* system of
units works as-is, and `cdarcy = 1`. What its docstring claims is that a
*non*-coherent (practical) system also works, given

    C = u_k u_p u_t / (u_mu u_L**2)    and    u_q = u_L**2 / u_t

writing `u_x` for the SI size of the unit chosen for quantity `x`. That is what
is tested here: one and the same reservoir, posed in each of the four
systems below, must yield one and the same answer. Nothing else pins the *value* of the
constant (as opposed to its placement) -- a regression value cannot.

NB: it is `u_q = u_L**2 / u_t`, not `u_L**3 / u_t`, because the model is 2D
*areal*: its rates are per unit thickness. Were that power wrong, or the
constant misplaced (in the accumulation term, say, or omitted from the well
index), the systems would disagree -- as they do, by 30-200%, if you perturb
either the exponent or `C` below.
"""

import numpy as np
import pytest

from TPFA_ResSim import ResSim, Wells

# The units' SI sizes -- whence the constants under test
DAY  = 86400.       # s
HOUR = 3600.        # s
BAR  = 1e5          # Pa
ATM  = 101325.      # Pa
PSI  = 6894.757     # Pa
FT   = 0.3048       # m
CM   = 0.01         # m
MD   = 9.869233e-16 # m² (i.e. 1 darcy = .9869233 µm²)
CP   = 1e-3         # Pa·s


def cdarcy_of(u_k, u_p, u_t, u_mu, u_L):
    """`ResSim.cdarcy` for a system, from its units' SI sizes."""
    return u_k * u_p * u_t / (u_mu * u_L**2)


def test_the_documented_constants():
    """The table in the `ResSim.cdarcy` docstring."""
    assert cdarcy_of(1, 1, 1, 1, 1) == 1
    assert np.isclose(cdarcy_of(MD, BAR, DAY, CP, 1.0), 0.008527, rtol=1e-4), \
        "metric, i.e. ECLIPSE's CDARCY"
    assert np.isclose(cdarcy_of(MD, PSI, DAY, CP, FT), 0.006328, rtol=1e-4)
    # Exactly 3.6, the darcy being *defined* in cm/s/atm/cP (so mD/hour ⇒ 3.6)
    assert np.isclose(cdarcy_of(MD, ATM, HOUR, CP, CM), 3.6, rtol=1e-6)
    assert ResSim().cdarcy == 1, "The default is the coherent one"


# One reservoir, in SI. Two phases moving, storage (`ct > 0`), heterogeneous
# and anisotropic rock, a rate-controlled injector and a BHP-controlled
# producer -- so that every term, and the well model, takes part.
N = 12
_rand = np.random.default_rng(4).uniform(0.5, 2.0, (2, N, N))
SI = dict(L=700.0, k=100 * MD, mu=1 * CP, ct=1e-4 / BAR,
          q=200 / DAY, p=250 * BAR, rw=0.1, dt=30 * DAY)  # fmt: skip


def in_units(u_k, u_p, u_t, u_mu, u_L, *, u_q=None, cdarcy=None):
    """Re-express `SI` in the given system, and simulate.

    Returns `(S, P, rates)`, all three converted back to SI, so that the
    systems are directly comparable. The overrides are for
    `test_the_claims_are_falsifiable`, and should otherwise be left alone.
    """
    u_q = u_L**2 / u_t if u_q is None else u_q  # NB: *areal*, ref the docstring
    model = ResSim(
        Nx=N, Ny=N,
        cdarcy=cdarcy_of(u_k, u_p, u_t, u_mu, u_L) if cdarcy is None else cdarcy,
        Lx=SI["L"]/u_L, Ly=SI["L"]/u_L, ct=SI["ct"]*u_p,
        vw=SI["mu"]/u_mu, vo=2*SI["mu"]/u_mu,
        K=SI["k"]/u_k * _rand, por=0.2*np.ones((N, N)), swc=0.1, sor=0.1,
        wells=[dict(xy=[0, 0], rate=+SI["q"]/u_q, rw=SI["rw"]/u_L),
               dict(xy=[SI["L"]/u_L]*2, bhp=0.9*SI["p"]/u_p, rw=SI["rw"]/u_L)],
    )  # fmt: skip
    S, P = model.sim(SI["dt"]/u_t, 8, np.full(model.Nxy, model.swc),
                     np.full(model.Nxy, SI["p"]/u_p), pbar=False)  # fmt: skip
    return S, P * u_p, model.wells.actual_rates * u_q


SYSTEMS = dict(
    si     = (1.0, 1.0, 1.0,  1.0, 1.0),  # m,  s,    Pa,  m²,  Pa·s
    metric = (MD,  BAR, DAY,  CP,  1.0),  # m,  day,  bar, mD,  cP
    field  = (MD,  PSI, DAY,  CP,  FT),   # ft, day,  psi, mD,  cP
    lab    = (MD,  ATM, HOUR, CP,  CM),   # cm, hour, atm, mD,  cP
)  # fmt: skip


@pytest.mark.parametrize("system", [s for s in SYSTEMS if s != "si"])
def test_agrees_with_coherent_SI(system):
    """The same reservoir, posed practically and coherently, runs the same."""
    S0, P0, q0 = in_units(*SYSTEMS["si"])
    S1, P1, q1 = in_units(*SYSTEMS[system])

    assert np.allclose(S1, S0, rtol=1e-8, atol=1e-10), "saturation"
    assert np.allclose(P1, P0, rtol=1e-8), "pressure"
    # The BHP-controlled producer found the same rate (its `u_q` undone)
    assert np.allclose(q1, q0, rtol=1e-8), "realized rates"
    # Sanity: the run is actually doing something
    assert S0.max() > 0.5, "The water front should have moved"
    assert q0[1].max() < 0, "The producer should produce"


@pytest.mark.parametrize("wrong", ["exponent", "constant"])
def test_the_claims_are_falsifiable(wrong):
    """Perturb the rate-unit exponent, or the constant: agreement must break.

    I.e. the test above does not pass vacuously: both halves of the pair above
    -- the constant, and the rate-unit exponent -- are load-bearing.
    """
    u_k, u_p, u_t, u_mu, u_L = SYSTEMS["field"]
    override = (
        # Rates as u_L**3/u_t (volumetric), which the *areal* model does not use
        dict(u_q=u_L**3 / u_t) if wrong == "exponent" else
        # ECLIPSE's FIELD `CDARCY`, which goes with bbl/day -- a rate unit that
        # an areal model cannot have (its volumes come from the pore volume)
        dict(cdarcy=0.001127)
    )  # fmt: skip
    _, P0, _ = in_units(*SYSTEMS["si"])
    _, P1, _ = in_units(u_k, u_p, u_t, u_mu, u_L, **override)
    assert not np.allclose(P1, P0, rtol=1e-3), "Should NOT agree"


def test_the_default_is_untouched():
    """`cdarcy` is opt-in: the results at `cdarcy = 1` must be as they were."""
    def run(**kwargs):
        model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, ct=0.1, **kwargs,
                       wells=Wells(xy=[[0, 0], [1, 1]], rates=[[1], [-1]]))  # fmt: skip
        return model.sim(0.05, 5, np.zeros(model.Nxy), pbar=False)

    S0, P0 = run()
    S1, P1 = run(cdarcy=1.0)
    assert (S1 == S0).all() and (P1 == P0).all()
