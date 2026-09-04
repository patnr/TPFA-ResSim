"""Tests of the tangent linear model, `TPFA_ResSim.tlm`.

The TLM is hand-derived, so the test that matters is against a finite
difference of the forward model -- on configurations that exercise each of
its terms: heterogeneous `K` (the harmonic averaging), irreducible saturations
and a viscosity contrast (the mobilities), compressibility (the accumulation
and storage terms, through which `P0` matters), BHP-controlled wells (the
well model in the system and the realized rates), and the pinned,
incompressible pressure system. The tapes are of the *unperturbed* run, as an
adjoint's would be.

Central differences with `eps = 1e-6` on a smooth map should agree to
~`1e-8`; the tolerance below is a hundred times that. The models solve their
pressure directly (`cached_precond=False`) so that the iterative solver's
tolerance (`1e-10`, amplified a million-fold by `eps`) does not enter the
comparison -- the TLM itself is indifferent (it factorizes its own system).
"""

import numpy as np
import pytest

from TPFA_ResSim import ResSim
from TPFA_ResSim.tlm import face_operators, fractional_flow, linearize, tlm, tlm_step

n = 12
dt = .0337  # not a round number, lest `dt * 1/CFL` land on an integer
nSteps = 3
eps = 1e-6
rng = np.random.default_rng(1)


def K_het():
    return np.exp(rng.standard_normal((n, n)))


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
}


def make_case(name):
    model = ResSim(Nx=n, Ny=n, cached_precond=False, **configs[name])
    N = model.Nxy
    S0 = .2 + .6 * rng.random(N)  # interior, away from the kinks at swc, 1-sor
    P0 = 1 + rng.random(N)
    dS0, dP0 = rng.standard_normal((2, N))
    return model, S0, P0, dS0, dP0


@pytest.mark.parametrize("name", configs)
def test_against_finite_difference(name):
    model, S0, P0, dS0, dP0 = make_case(name)
    SS, PP = model.sim(dt, nSteps, S0, P0, pbar=False)
    dSS, dPP = tlm(model, dt, SS, PP, dS0, dP0)

    SSp, PPp = model.sim(dt, nSteps, S0 + eps*dS0, P0 + eps*dP0, pbar=False)
    SSm, PPm = model.sim(dt, nSteps, S0 - eps*dS0, P0 - eps*dP0, pbar=False)
    fdS, fdP = (SSp - SSm) / (2*eps), (PPp - PPm) / (2*eps)

    assert abs(dSS).max() > .1 and abs(dPP).max() > .1  # non-trivial
    assert abs(fdS - dSS).max() < 1e-6 * abs(dSS).max()
    assert abs(fdP - dPP).max() < 1e-6 * abs(dPP).max()


@pytest.mark.parametrize("name", configs)
def test_linearize_reproduces_the_step(name):
    """The recomputed forward step (which the tape linearizes about) is `sim`'s."""
    model, S0, P0, _, _ = make_case(name)
    SS, PP = model.sim(dt, nSteps, S0, P0, pbar=False)
    for k in range(nSteps):
        tape = linearize(model, dt, SS[k], PP[k], k)
        assert np.allclose(tape.S1, SS[k + 1], rtol=0, atol=1e-13)
        assert np.allclose(tape.P1, PP[k + 1], rtol=0, atol=1e-12)


def test_face_operators_recast_the_assemblies():
    """The face-operator forms of `TPFA`'s fluxes and of `upwind_diff` match them."""
    model, S0, P0, _, _ = make_case("bhp_compressible")
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
    model, S0, P0, dS0, dP0 = make_case("incompressible")
    tape = linearize(model, dt, S0, P0, 0)
    dS1, dP1 = tlm_step(tape, dS0, dP0)
    dS1b, dP1b = tlm_step(tape, dS0, 0 * dP0)
    assert np.array_equal(dS1, dS1b) and np.array_equal(dP1, dP1b)


def test_linearity():
    """`tlm_step` is linear in `(dS, dP)`, and does not touch its inputs."""
    model, S0, P0, dS0, dP0 = make_case("bhp_compressible")
    tape = linearize(model, dt, S0, P0, 0)
    dS0_, dP0_ = dS0.copy(), dP0.copy()
    a, b = 2.5, -1.5
    dS1, dP1 = tlm_step(tape, dS0, dP0)
    dS2, dP2 = tlm_step(tape, dS0[::-1], dP0[::-1])
    dS3, dP3 = tlm_step(tape, a*dS0 + b*dS0[::-1], a*dP0 + b*dP0[::-1])
    assert np.allclose(dS3, a*dS1 + b*dS2) and np.allclose(dP3, a*dP1 + b*dP2)
    assert np.array_equal(dS0, dS0_) and np.array_equal(dP0, dP0_)
