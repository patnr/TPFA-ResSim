"""Tests of the tangent linear model and its adjoint, `TPFA_ResSim.tlm`.

Both are hand-derived, so the tests that matter are (i) the TLM against a
finite difference of the forward model, and (ii) the adjoint against the TLM
by the dot-product test, $ ⟨ M δx, ȳ ⟩ = ⟨ δx, M^T ȳ ⟩ $, which holds to
round-off; (iii) then closes the loop, checking `adjoint`'s gradient of an
objective against a finite difference of the objective. The configurations
exercise each term: heterogeneous `K` (the harmonic averaging), irreducible
saturations and a viscosity contrast (the mobilities), compressibility (the
accumulation and storage terms, through which `P0` matters), BHP-controlled
wells (the well model in the system and the realized rates), the pinned,
incompressible pressure system, and a 1D row (`Ny = 1`: no y-faces). The tapes
are of the *unperturbed* run, as an adjoint's would be.

Central differences with `eps = 1e-6` on a smooth map should agree to
~`1e-8`; the tolerance below is a hundred times that. The models solve their
pressure directly (`cached_precond=False`) so that the iterative solver's
tolerance (`1e-10`, amplified a million-fold by `eps`) does not enter the
comparison -- the TLM itself is indifferent (it factorizes its own system).
"""

import numpy as np
import pytest

from TPFA_ResSim import ResSim
from TPFA_ResSim.tlm import (
    adj_step, adjoint, face_operators, fractional_flow, linearize, tlm, tlm_step,
)

n = 12
dt = .0337  # not a round number, lest `dt * 1/CFL` land on an integer
nSteps = 3
eps = 1e-6
rng = np.random.default_rng(1)


def K_het(shape=(n, n)):
    return np.exp(rng.standard_normal(shape))


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
}


def make_case(name):
    kws: dict = {**dict(Nx=n, Ny=n), **configs[name]}
    model = ResSim(**kws, cached_precond=False)
    N = model.Nxy
    S0 = .2 + .6 * rng.random(N)  # interior, away from the kinks at swc, 1-sor
    P0 = 1 + rng.random(N)
    dS0, dP0 = rng.standard_normal((2, N))
    dlogK = rng.standard_normal(model.K.shape)
    return model, S0, P0, dS0, dP0, dlogK


def perturbed_sim(model, S0, P0, dS0, dP0, dlogK, eps):
    """`sim` from the perturbed initial state and permeability."""
    K = model.K.copy()
    model.K = K * np.exp(eps * dlogK)
    SS, PP = model.sim(dt, nSteps, S0 + eps*dS0, P0 + eps*dP0, pbar=False)
    model.K = K
    return SS, PP


@pytest.mark.parametrize("name", configs)
def test_tlm_against_finite_difference(name):
    model, S0, P0, dS0, dP0, dlogK = make_case(name)
    SS, PP = model.sim(dt, nSteps, S0, P0, pbar=False)
    dSS, dPP = tlm(model, dt, SS, PP, dS0, dP0, dlogK)

    SSp, PPp = perturbed_sim(model, S0, P0, dS0, dP0, dlogK, +eps)
    SSm, PPm = perturbed_sim(model, S0, P0, dS0, dP0, dlogK, -eps)
    fdS, fdP = (SSp - SSm) / (2*eps), (PPp - PPm) / (2*eps)

    assert abs(dSS).max() > .1 and abs(dPP).max() > .1  # non-trivial
    assert abs(fdS - dSS).max() < 1e-6 * abs(dSS).max()
    assert abs(fdP - dPP).max() < 1e-6 * abs(dPP).max()


@pytest.mark.parametrize("name", configs)
def test_adjoint_dot_product(name):
    """`adj_step` is the transpose of `tlm_step`, to round-off."""
    model, S0, P0, dS0, dP0, dlogK = make_case(name)
    tape = linearize(model, dt, S0, P0, 0)
    aS1, aP1 = rng.standard_normal((2, model.Nxy))
    dS1, dP1 = tlm_step(tape, dS0, dP0, dlogK)
    aS0, aP0, alogK = adj_step(tape, aS1, aP1)
    lhs = dS1 @ aS1 + dP1 @ aP1
    rhs = dS0 @ aS0 + dP0 @ aP0 + (dlogK * alogK).sum()
    assert abs(lhs - rhs) < 1e-12 * abs(lhs)
    # Every input reaches every output (else the test above proves less)
    assert abs(aS0).max() > 0 and abs(alogK).max() > 0
    assert (abs(aP0).max() > 0) == (model.ct > 0)


@pytest.mark.parametrize("name", configs)
def test_adjoint_gradient_against_finite_difference(name):
    """`adjoint`'s gradient of a (random, linear) objective of the whole
    trajectory, in a random direction of `(S0, P0, log K)`."""
    model, S0, P0, dS0, dP0, dlogK = make_case(name)
    SS, PP = model.sim(dt, nSteps, S0, P0, pbar=False)
    wS, wP = rng.standard_normal((2, *SS.shape))
    def J(SS, PP): return (wS * SS).sum() + (wP * PP).sum()
    grad = adjoint(model, dt, SS, PP, wS, wP)
    directional = grad.S0 @ dS0 + grad.P0 @ dP0 + (grad.logK * dlogK).sum()
    Jp = J(*perturbed_sim(model, S0, P0, dS0, dP0, dlogK, +eps))
    Jm = J(*perturbed_sim(model, S0, P0, dS0, dP0, dlogK, -eps))
    assert abs((Jp - Jm) / (2*eps) - directional) < 1e-6 * abs(directional)
    assert grad.logK.shape == model.K.shape


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


def test_incompressible_pressure_ignores_P0():
    """With `ct == 0` and rate control only, `P1` is a function of `S` alone."""
    model, S0, P0, dS0, dP0, _ = make_case("incompressible")
    tape = linearize(model, dt, S0, P0, 0)
    dS1, dP1 = tlm_step(tape, dS0, dP0)
    dS1b, dP1b = tlm_step(tape, dS0, 0 * dP0)
    assert np.array_equal(dS1, dS1b) and np.array_equal(dP1, dP1b)
    assert np.array_equal(adj_step(tape, dS0, dP0)[1], 0 * P0)


def test_linearity():
    """`tlm_step` is linear in `(dS, dP, dlogK)`, and does not touch its inputs."""
    model, S0, P0, dS0, dP0, dlogK = make_case("bhp_compressible")
    tape = linearize(model, dt, S0, P0, 0)
    dS0_, dP0_ = dS0.copy(), dP0.copy()
    a, b = 2.5, -1.5
    dS1, dP1 = tlm_step(tape, dS0, dP0, dlogK)
    dS2, dP2 = tlm_step(tape, dS0[::-1], dP0[::-1], -dlogK)
    dS3, dP3 = tlm_step(tape, a*dS0 + b*dS0[::-1], a*dP0 + b*dP0[::-1], (a - b)*dlogK)
    assert np.allclose(dS3, a*dS1 + b*dS2) and np.allclose(dP3, a*dP1 + b*dP2)
    assert np.array_equal(dS0, dS0_) and np.array_equal(dP0, dP0_)


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
