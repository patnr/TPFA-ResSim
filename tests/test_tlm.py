"""Tests of the adjoint, `TPFA_ResSim.tlm`.

It is hand-derived, so the test that matters is the gradient it returns, of an
objective of the whole trajectory, against a finite difference of that
objective -- in a random direction of `(S0, P0, log K, bhp)`, on
configurations that exercise each term: heterogeneous `K` (the harmonic
averaging), irreducible saturations and a viscosity contrast (the mobilities),
compressibility (the accumulation and storage terms, through which `P0`
matters), BHP-controlled wells (the well model in the system, the realized
rates, and the BHP *schedule* as a parameter -- made a full `(nComp, nSteps)`
one, so that each step's entry of the gradient is checked), the pinned,
incompressible pressure system, a 1D row (`Ny = 1`: no y-faces), inactive
cells (`active`: omitted faces, identity rows, and the pin moved off cell 0),
and an aquifer (BHP completions on the boundary, ref `aquifer_WI`, supplying a
lone producer -- so its gradient is the sensitivity to the aquifer pressure).
The tapes are of the *unperturbed* run, as they are in use.

Central differences with `eps = 1e-5` on a smooth map should agree to
~`1e-8` (the truncation error grows as `eps²`, the round-off of the differenced
objectives as `1/eps`, and this is about where they balance); the tolerance
below is a hundred times that. The models solve their
pressure directly (`cached_precond=False`) so that the iterative solver's
tolerance (`1e-10`, amplified a million-fold by `eps`) does not enter the
comparison -- the adjoint itself is indifferent (it factorizes its own system).

(The tangent linear model that `adj_step` was transposed from, and the
dot-product test that bound the two to round-off, are in the git history.)
"""

import numpy as np
import pytest

from TPFA_ResSim import ResSim
from TPFA_ResSim.tlm import adj_step, adjoint, face_operators, fractional_flow, linearize

n = 12
dt = .0337  # not a round number, lest `dt * 1/CFL` land on an integer
nSteps = 3
eps = 1e-5
rng = np.random.default_rng(1)


def K_het(shape=(n, n)):
    return np.exp(rng.standard_normal(shape))


def mask():
    """A hole, a fault reaching from the top down to the row above cell 0, and
    that cell itself inactive (so that the pin moves)."""
    active = np.ones((n, n), bool)
    active[4:8, 4:8] = False
    active[9, 1:] = False
    active[0, 0] = False
    return active


configs: dict = {
    "incompressible": dict(
        K=K_het(), swc=.1, sor=.2, vo=3,
        wells=[dict(xy=[0, 0], rate=1), dict(xy=[1, 1], rate=-1)],
    ),
    "compressible": dict(
        K=K_het(), ct=.1, vw=2,
        wells=[dict(xy=[0, 0], rate=1), dict(xy=[1, 1], rate=-.4), dict(xy=[.5, .9], rate=-.3)],
    ),
    "bhp_compressible": dict(
        K=K_het(), ct=.05,
        wells=[dict(xy=[0, 0], bhp=3, rw=1e-3), dict(xy=[1, 1], bhp=0, rw=1e-3),
               dict(xy=[.5, .5], rate=-.2)],
    ),
    "bhp_incompressible": dict(
        K=K_het(),
        wells=[dict(xy=[0, 0], bhp=3, rw=1e-3), dict(xy=[1, 1], bhp=0, rw=1e-3)],
    ),
    "row_1d": dict(
        Nx=3*n, Ny=1, K=K_het((3*n, 1)), ct=.1,
        wells=[dict(xy=[0, 0], rate=1), dict(xy=[1, 1], bhp=0, rw=1e-3)],
    ),
    "inactive_incompressible": dict(
        K=K_het(), active=mask(), swc=.1, sor=.1, vo=2,
        wells=[dict(xy=[.1, 0], rate=1), dict(xy=[.6, 1], rate=-1),
               dict(xy=[1, 0], rate=.5), dict(xy=[1, 1], rate=-.5)],
    ),
    "inactive_bhp_compressible": dict(
        K=K_het(), active=mask(), ct=.1,
        wells=[dict(xy=[.1, 0], bhp=3, rw=1e-3), dict(xy=[.6, 1], rate=-1),
               dict(xy=[1, 0], rate=.5), dict(xy=[1, 1], bhp=0, rw=1e-3)],
    ),
    "aquifer_incompressible": dict(
        K=K_het(), active=mask(), vo=2,
        wells=[dict(xy=[[0, .4], [0, .5], [0, .6]], aquifer=True, bhp=2),
               dict(xy=[1, 1], rate=-1)],
    ),
}


def make_case(name):
    kws: dict = {**dict(Nx=n, Ny=n), **configs[name]}
    model = ResSim(**kws, cached_precond=False)
    N = model.Nxy
    S0 = .2 + .6 * rng.random(N)  # interior, away from the kinks at swc, 1-sor
    P0 = 1 + rng.random(N)
    dS0, dP0 = rng.standard_normal((2, N))
    dlogK = rng.standard_normal(model.K.shape)
    # The BHP schedule, as a full `(nComp, nSteps)` one (perturbed where finite)
    if model.wells.bhp is not None:
        model.wells.bhp = np.repeat(model.wells.bhp, nSteps, axis=1)
        dbhp = rng.standard_normal((model.nComp, nSteps))
        dbhp = np.where(np.isfinite(model.wells.bhp), dbhp, 0)
    else:
        dbhp = np.zeros((model.nComp, nSteps))
    return model, S0, P0, dS0, dP0, dlogK, dbhp


def perturbed_sim(model, S0, P0, dS0, dP0, dlogK, dbhp, eps):
    """`sim` from the perturbed initial state, permeability and BHP schedule."""
    K, bhp = model.K.copy(), model.wells.bhp
    model.K = K * np.exp(eps * dlogK)
    if bhp is not None:
        model.wells.bhp = bhp + eps * dbhp
    SS, PP = model.sim(dt, nSteps, S0 + eps*dS0, P0 + eps*dP0, pbar=False)
    model.K, model.wells.bhp = K, bhp
    return SS, PP


@pytest.mark.parametrize("name", configs)
def test_gradient_against_finite_difference(name):
    """`adjoint`'s gradient of a (random, linear) objective of the whole
    trajectory, in a random direction of `(S0, P0, log K, bhp)`."""
    model, S0, P0, dS0, dP0, dlogK, dbhp = make_case(name)
    SS, PP = model.sim(dt, nSteps, S0, P0, pbar=False)
    wS, wP = rng.standard_normal((2, *SS.shape))
    def J(SS, PP): return (wS * SS).sum() + (wP * PP).sum()
    grad = adjoint(model, dt, SS, PP, wS, wP)
    directional = (grad.S0 @ dS0 + grad.P0 @ dP0
                   + (grad.logK * dlogK).sum() + (grad.bhp * dbhp).sum())
    Jp = J(*perturbed_sim(model, S0, P0, dS0, dP0, dlogK, dbhp, +eps))
    Jm = J(*perturbed_sim(model, S0, P0, dS0, dP0, dlogK, dbhp, -eps))
    assert abs((Jp - Jm) / (2*eps) - directional) < 1e-6 * abs(directional)
    # Every parameter contributes (else the test above proves less)
    assert abs(grad.S0).max() > 0 and abs(grad.logK).max() > 0
    act = model.active.ravel()
    assert (abs(grad.P0 - wP[0])[act].max() > 0) == (model.ct > 0)  # beyond the direct seed
    # An inactive cell's state is its initial state throughout, and it has no faces
    assert np.allclose(grad.S0[~act], wS[:, ~act].sum(0))
    assert np.allclose(grad.P0[~act], wP[:, ~act].sum(0))
    assert not grad.logK[:, ~model.active].any()
    assert grad.logK.shape == model.K.shape
    # ... the BHP one at every step of every BHP-controlled completion, and nowhere else
    assert grad.bhp.shape == (model.nComp, nSteps)
    is_bhp = dbhp.any(axis=1)
    assert (grad.bhp[is_bhp] != 0).all() and not grad.bhp[~is_bhp].any()


@pytest.mark.parametrize("name", configs)
def test_linearize_reproduces_the_step(name):
    """The recomputed forward step (which the tape linearizes about) is `sim`'s."""
    model, S0, P0, *_ = make_case(name)
    SS, PP = model.sim(dt, nSteps, S0, P0, pbar=False)
    for k in range(nSteps):
        tape = linearize(model, dt, SS[k], PP[k], k)
        assert np.allclose(tape.S1, SS[k + 1], rtol=0, atol=1e-13)
        assert np.allclose(tape.P1, PP[k + 1], rtol=0, atol=1e-12)


def test_face_operators_recast_the_assemblies():
    """The face-operator forms of `TPFA`'s fluxes and of `upwind_diff` match them."""
    model, S0, P0, *_ = make_case("bhp_compressible")
    lo, hi, Grad, Sum, g = face_operators(model)
    tape = linearize(model, dt, S0, P0, 0)
    # The fluxes: `TPFA`'s, on the interior faces, x-faces first
    model.assemble_wells(S0, P0, 0)
    _, VV = model.pressure_step(S0, P0, dt)
    model.realize_bhp(tape.P1)
    V = np.r_[VV.x[1:-1, :].ravel(), VV.y[:, 1:-1].ravel()]
    assert np.allclose(tape.V, V, rtol=0, atol=1e-14)
    assert np.allclose(Grad.T @ V, model._Q - model.storage_rate(VV))
    # The transport operator: `A_up @ fw == -div(V * fw_upwind) + Q⁻ * fw`
    fw, _ = fractional_flow(model, S0)
    A_up = model.upwind_diff(VV)
    rhs = -Grad.T @ (tape.V * (tape.Up @ fw)) + tape.Q.clip(max=0) * fw
    assert np.allclose(A_up @ fw, rhs, rtol=0, atol=1e-14)


def test_adj_step_is_linear_and_leaves_its_inputs():
    model, S0, P0, aS1, aP1, *_ = make_case("bhp_compressible")
    tape = linearize(model, dt, S0, P0, 0)
    aS1_, aP1_ = aS1.copy(), aP1.copy()
    a, b = 2.5, -1.5
    out1 = adj_step(tape, aS1, aP1)
    out2 = adj_step(tape, aS1[::-1], aP1[::-1])
    out3 = adj_step(tape, a*aS1 + b*aS1[::-1], a*aP1 + b*aP1[::-1])
    for x1, x2, x3 in zip(out1, out2, out3):
        assert np.allclose(x3, a*x1 + b*x2)
    assert np.array_equal(aS1, aS1_) and np.array_equal(aP1, aP1_)


def test_incompressible_pressure_ignores_P0():
    """With `ct == 0` and rate control only, `P1` is a function of `S` alone."""
    model, S0, P0, aS1, aP1, *_ = make_case("incompressible")
    tape = linearize(model, dt, S0, P0, 0)
    assert np.array_equal(adj_step(tape, aS1, aP1)[1], 0 * P0)


def test_linearize_leaves_the_reports_alone():
    """`linearize` mutates the model as a step does (`_Q`, `_wells_now`, `_pLU`),
    but `sim`'s reports, `actual_rates`/`actual_bhp`, survive an adjoint sweep."""
    model, S0, P0, *_ = make_case("bhp_compressible")
    model.cached_precond = True
    SS, PP = model.sim(dt, nSteps, S0, P0, pbar=False)
    rates, bhp = model.wells.actual_rates.copy(), model.wells.actual_bhp.copy()
    adjoint(model, dt, SS, PP, np.ones_like(SS))
    assert np.array_equal(model.wells.actual_rates, rates)
    assert np.array_equal(model.wells.actual_bhp, bhp, equal_nan=True)
    # ... and the model still simulates the same trajectory afterwards
    SS2, PP2 = model.sim(dt, nSteps, S0, P0, pbar=False)
    assert np.allclose(SS2, SS, atol=1e-10) and np.allclose(PP2, PP, atol=1e-9)
