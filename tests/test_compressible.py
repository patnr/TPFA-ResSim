"""Tests of the slightly compressible model (`ct > 0`).

Unlike the examples (run by `test_examples`), there is no Matlab reference to
compare with, and no figures. Instead we verify structural properties: exact
discrete material balance, pressure decline under depletion, recovery of the
incompressible limit, and the consistency of the storage term shared by the
pressure and transport equations.
"""

import numpy as np
import pytest

from TPFA_ResSim import ResSim

nSteps = 10
dt = .05


def test_material_balance():
    """Summing the rows of the (conservative) system gives, exactly,
    $ c_t \\sum_{ij} φ_{ij} h^2 (p^{n+1}_{ij} - p^n_{ij}) = Δt \\sum Q $."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=1e-2,
                   inj_xy=[[0, 0]], inj_rates=[[.5]],
                   prd_xy=[[1, 1]], prd_rates=[[1]])
    water_sat0 = np.zeros(model.Nxy)
    SS, PP = model.sim(dt, nSteps, water_sat0, pbar=False)

    pv = model.h2 * model.por.ravel()  # pore volumes
    Q_total = .5 - 1                   # inj - prd
    for k in range(nSteps):
        storage = model.ct * (pv * (PP[k+1] - PP[k])).sum()
        assert np.isclose(storage, dt * Q_total)


def test_depletion():
    """Production without injection (impossible if incompressible):
    the average pressure declines monotonically."""
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=1e-2,
                   inj_xy=[[0, 0]], inj_rates=[[0]],
                   prd_xy=[[1, 1]], prd_rates=[[1]])
    water_sat0 = np.zeros(model.Nxy)
    p0 = np.ones(model.Nxy)
    SS, PP = model.sim(dt, nSteps, water_sat0, p0=p0, pbar=False)

    means = PP.mean(axis=1)
    assert np.all(np.diff(means) < 0)


def test_incompressible_limit():
    """With balanced wells and tiny `ct`, the pressure *differences*
    (pressure is only defined up to a constant when incompressible)
    should approach those of the `ct = 0` model."""
    kwargs: dict = dict(Lx=1, Ly=1, Nx=20, Ny=20,
                  inj_xy=[[0, 0]], inj_rates=[[1]],
                  prd_xy=[[1, 1]], prd_rates=[[1]])
    water_sat0 = np.zeros(20 * 20)

    model0 = ResSim(**kwargs)
    SS0, PP0 = model0.sim(dt, nSteps, water_sat0, pbar=False)

    model1 = ResSim(ct=1e-8, **kwargs)
    SS1, PP1 = model1.sim(dt, nSteps, water_sat0, pbar=False)

    center = lambda P: P - P.mean(axis=1, keepdims=True)
    assert np.allclose(center(PP0[1:]), center(PP1[1:]), atol=1e-4)
    # Saturations differ a bit more: tiny flux differences can flip the
    # (integer) number of CFL-derived sub-steps of the explicit scheme.
    assert np.allclose(SS0, SS1, atol=1e-3)


def test_storage_rate():
    """`storage_rate` is the accumulation term of the pressure eqn.

    I.e. the two equations agree on how much volume went into storage,
    which is what keeps the transport step consistent (ref `ResSim.ct`).
    """
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=1e-2,
                   inj_xy=[[0, 0]], inj_rates=[[.5]],
                   prd_xy=[[1, 1]], prd_rates=[[1]])
    water_sat0 = np.zeros(model.Nxy)
    p_prev = np.ones(model.Nxy)
    model._set_Q(water_sat0, 0)  # (as `time_stepper` does)
    P, V = model.pressure_step(water_sat0, p_prev, dt)

    accum = model.por.ravel() * model.ct * model.h2 / dt
    assert np.allclose(model.storage_rate(V), accum * (P.ravel() - p_prev))
    # Globally, the storage is the well imbalance (cf. `test_material_balance`)
    assert np.isclose(model.storage_rate(V).sum(), .5 - 1)


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("ct", [1e-2, 1])
@pytest.mark.parametrize("vrr", [0, .5, 1, 2])
def test_single_phase_is_exact(implicit, ct, vrr):
    """A single-phase reservoir stays that way, whatever the wells do.

    With only water present, `s = 1` must persist: the produced volume comes
    from expansion (storage), not from the (nonexistent) oil. And, when nothing
    is injected, `s = 0` likewise persists. Both are exact fixed points of the
    transport step *because* it charges the storage to the phases in proportion
    to their saturation. Neglecting that term would instead drain `s` by the
    voidage (and, at the injector, run away).
    """
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=ct,
                   inj_xy=[[0, 0]], inj_rates=[[vrr]],
                   prd_xy=[[1, 1]], prd_rates=[[1]])
    p0 = 10*np.ones(model.Nxy)
    for s in ([0, 1] if vrr == 0 else [1]):  # only water is injected
        SS, PP = model.sim(dt, nSteps, s*np.ones(model.Nxy), p0=p0,
                           pbar=False, implicit=implicit)
        assert np.allclose(SS, s, atol=1e-12)
