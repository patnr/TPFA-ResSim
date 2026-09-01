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
    P0 = np.ones(model.Nxy)
    SS, PP = model.sim(dt, nSteps, water_sat0, P0=P0, pbar=False)

    means = PP.mean(axis=1)
    assert np.all(np.diff(means) < 0)


def test_incompressible_limit():
    """With balanced wells, the `ct = 0` model is recovered as `ct → 0`,
    at first order. Compare the pressure *differences*, since pressure is
    only defined up to a constant when incompressible.
    """
    kwargs: dict = dict(Lx=1, Ly=1, Nx=20, Ny=20,
                  inj_xy=[[0, 0]], inj_rates=[[1]],
                  prd_xy=[[1, 1]], prd_rates=[[1]])
    water_sat0 = np.zeros(20 * 20)
    # NB: not the module-level `dt = .05`, for which `dt * estimate_1CFL` is
    # *exactly* 60 here, leaving the (integer) sub-step count of the explicit
    # scheme to be settled by the last bits of the flux computation. It then
    # differs between the two models -- and between platforms -- imposing a
    # `ct`-independent floor of ~1e-4 on the discrepancies below.
    dt = .0505

    model0 = ResSim(**kwargs)
    SS0, PP0 = model0.sim(dt, nSteps, water_sat0, pbar=False)

    center = lambda P: P - P.mean(axis=1, keepdims=True)
    for ct in [1e-8, 1e-6, 1e-4]:
        SS1, PP1 = ResSim(ct=ct, **kwargs).sim(dt, nSteps, water_sat0, pbar=False)
        # Both are O(ct) (with a coefficient < 2), so the smallest `ct`
        # would fail were there any floor, as with `dt = .05`.
        assert np.allclose(center(PP0[1:]), center(PP1[1:]), rtol=0, atol=10*ct)
        assert np.allclose(SS0, SS1, rtol=0, atol=10*ct)


def test_storage_rate():
    """`storage_rate` is the accumulation term of the pressure eqn.

    I.e. the two equations agree on how much volume went into storage,
    which is what keeps the transport step consistent (ref `ResSim.ct`).
    """
    model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20, ct=1e-2,
                   inj_xy=[[0, 0]], inj_rates=[[.5]],
                   prd_xy=[[1, 1]], prd_rates=[[1]])
    water_sat0 = np.zeros(model.Nxy)
    P0 = np.ones(model.Nxy)
    model.assemble_wells(water_sat0, 0)  # (as `time_stepper` does)
    P, V = model.pressure_step(water_sat0, P0, dt)

    accum = model.por.ravel() * model.ct * model.h2 / dt
    assert np.allclose(model.storage_rate(V), accum * (P - P0))
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
    P0 = 10*np.ones(model.Nxy)
    for s in ([0, 1] if vrr == 0 else [1]):  # only water is injected
        SS, PP = model.sim(dt, nSteps, s*np.ones(model.Nxy), P0=P0,
                           pbar=False, implicit=implicit)
        assert np.allclose(SS, s, atol=1e-12)
