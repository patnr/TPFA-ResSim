"""The tangent linear model (TLM) of `TPFA_ResSim.ResSim.time_stepper`, derived by hand.

I.e. the Jacobian of one time step, $ (s^n, p^n) ↦ (s^{n+1}, p^{n+1}) $, with
respect to the *state* -- applied to a perturbation, $ (δs, δp) $, rather than
formed (it is dense, through $A^{-1}$). The parameters (`K`, `por`, the well
controls) are held fixed: this is the linearization that gradients with respect
to the *initial* state, `S0` and `P0` of `TPFA_ResSim.ResSim.sim`, are made
of -- by running it in reverse, i.e. the adjoint, which is why it is written
the way it is (see below).

`linearize` recomputes one forward step -- from the state it is given, so the
trajectory that `sim` returns is all the record it needs (checkpointing) --
and returns a `Tape` of the intermediates. `tlm_step` then propagates a
perturbation through that step. `tlm` does both along a whole trajectory,
which the following checks against a finite difference:

>>> from TPFA_ResSim import ResSim
>>> model = ResSim(Lx=1, Ly=1, Nx=8, Ny=8, ct=.1, cached_precond=False, wells=[
...     dict(xy=[0, 0], rate=+1),
...     dict(xy=[1, 1], rate=-.5),
... ])
>>> rng = np.random.default_rng(3)
>>> S0 = .2 + .6 * rng.random(model.Nxy)
>>> P0 = rng.random(model.Nxy)
>>> dt, nSteps = .0337, 3
>>> SS, PP = model.sim(dt, nSteps, S0, P0, pbar=False)
>>> dS0, dP0 = rng.standard_normal((2, model.Nxy))
>>> dSS, dPP = tlm(model, dt, SS, PP, dS0, dP0)

>>> eps = 1e-6
>>> SSp, PPp = model.sim(dt, nSteps, S0 + eps*dS0, P0 + eps*dP0, pbar=False)
>>> SSm, PPm = model.sim(dt, nSteps, S0 - eps*dS0, P0 - eps*dP0, pbar=False)
>>> bool(np.abs((SSp - SSm) / (2*eps) - dSS).max() < 1e-6)
True
>>> bool(np.abs((PPp - PPm) / (2*eps) - dPP).max() < 1e-6)
True

(`cached_precond=False` merely spares the finite difference the noise of the
iterative solver's tolerance, $10^{-10}$, which `eps` would amplify a
million-fold; the TLM itself does not care.)

## Written to be reversed

`tlm_step` is a straight-line sequence of statements (the one loop being over
the transport sub-steps), each linear in the perturbations, and each of one of
three kinds:

- `y = M @ x`, with `M` a sparse matrix held by the tape (`Tape.Grad`,
  `Tape.Up`, ...);
- `y = a * x`, with `a` a vector (i.e. a diagonal matrix) held by the tape,
  or computed from it;
- `y = tape.solve(x)`, the pressure solve, $ A^{-1} x $.

The adjoint is therefore the same statements in *reverse* order, each
transposed -- `x̄ += M.T @ ȳ`, `x̄ += a * ȳ` -- with the pressure solve its own
transpose, $ A $ being symmetric (that is what makes the TPFA system SPD, ref
`TPFA_ResSim.ResSim.cached_precond`), so `Tape.solve` serves both directions
unchanged. Nothing else happens in there: no indexing gymnastics (the
5-diagonal assemblies of `TPFA_ResSim.ResSim.TPFA` and
`TPFA_ResSim.ResSim.upwind_diff` are recast on the face operators below, whose
transposes are just `.T`), no branching on the perturbations.

To that end the discretization is expressed on the **interior faces** (the
boundary ones carry no flux), numbered x-faces first, in C-order, as
`TX[1:-1, :]` and `TY[:, 1:-1]` of `TPFA` flatten. With `Grad` the
$ (n_F × N) $ difference operator, $ (∇p)_f = p_\\mathrm{lo} - p_\\mathrm{hi} $,
whose transpose is the divergence (cell $i$ gets its high faces minus its low
ones), the whole step reads

$$ A = ∇^T \\, \\mathrm{diag}(T) \\, ∇ + \\mathrm{diag}(a + w) \\,, \\qquad
   p^{n+1} = A^{-1} (Q + a \\, p^n + r) \\,, \\qquad
   v = T ⊙ ∇ p^{n+1} \\,, $$

with $T$ the transmissibilities (a function of the cell mobilities
$λ_t(s^n)$, ref `Tape.dT_dMt`), $a$ the accumulation term (`Tape.accum`) and
$w$, $r$ the well model of the BHP-controlled wells (`Tape.bhp_diag`, and the
matching right-hand side, both proportional to $λ_t$ in the well cells); then,
for each of the `nT` sub-steps,

$$ s ← s + \\frac{Δt}{n_T \\, |Ω|}
   \\Big( -∇^T \\big( v ⊙ \\mathrm{Up}(v) \\, f(s) \\big)
   + Q^- f(s) + Q^+ - s \\, \\mathrm{st} \\Big) \\,, $$

with `Up` the $ (n_F × N) $ upwind *selector* (1 in the column of the face's
upwind cell) and `st` the storage rate (0 if incompressible). `tlm_step`
differentiates exactly that, term by term.

## What is (and is not) differentiated

- The **state**, $ (s^n, p^n) $, and everything downstream of it: the
  mobilities, the transmissibilities, the pressure and the fluxes, the well
  model of a BHP-controlled well (its $ WI λ_t $ and the rate it realizes),
  the storage rate and the transport. If `ct == 0` and no well is on BHP
  control, $ p^{n+1} $ does not depend on $ p^n $ at all, so `dP` is
  simply dropped there (a gradient with respect to `P0` is then `0`).
- **Not** the parameters: `K`, `por`, `ct`, the well positions and
  specifications. These stay fixed, as they are in `sim`.
- **Not** the controls' dependence on the state: `well_controls` is assumed
  *open-loop* (the default). An override that feeds the state back is not
  seen -- the controls enter as constants.
- The **discrete decisions** are frozen at the linearization point: the
  sub-step count `nT` (a ceiling, ref `TPFA_ResSim.ResSim.estimate_1CFL`),
  the upwind directions and the signs of the well fluxes (the `clip`s of
  `upwind_diff`). They are piecewise constant, so this is the derivative
  almost everywhere; at a switch (a face with exactly zero flux, `nT` on an
  integer) the forward map has a kink, and a finite difference across it
  will disagree.
- Only the **explicit** transport scheme (`saturation_step_upwind`, the
  default). The implicit one has no TLM here (ref the "How to solve" section
  of the docs for why it is not used).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from TPFA_ResSim._repr import AlignedRepr

if TYPE_CHECKING:
    from TPFA_ResSim import ResSim


def face_operators(model: "ResSim") -> tuple:
    """The interior faces of the grid, and the sparse operators on them.

    Returns `(lo, hi, Grad, Sum, g)`:

    - `lo`, `hi`: the flat indices of the two cells each face separates
      (`lo` has the smaller index), shape `(nF,)`. The x-faces come first
      (`(Nx-1) * Ny` of them, in C-order), then the y-faces (`Nx * (Ny-1)`).
    - `Grad`: `(nF, Nxy)`, `(Grad @ p)[f] = p[lo] - p[hi]`, as `TPFA` computes
      the fluxes; `Grad.T` is the divergence (high faces minus low faces).
    - `Sum`: `(nF, 2*Nxy)`, summing over the face's two cells the *directional*
      component of a `(2, Nx, Ny)`-shaped field (flattened): the x-component
      for x-faces, the y-component for y-faces. This is how the transmissibility
      harmonically averages the permeabilities.
    - `g`: `(nF,)`, the geometric factor of the transmissibilities,
      $ 2 C h_y / h_x $ resp. $ 2 C h_x / h_y $, such that `T = g / (Sum @ (1/KM))`.

    >>> from TPFA_ResSim import ResSim
    >>> lo, hi, Grad, Sum, g = face_operators(ResSim(Nx=3, Ny=2))
    >>> lo, hi   # the 4 x-faces, then the 3 y-faces
    (array([0, 1, 2, 3, 0, 2, 4]), array([2, 3, 4, 5, 1, 3, 5]))
    >>> (Grad.T @ np.ones(7)).astype(int)   # #high faces - #low faces, per cell
    array([ 2,  0,  1, -1,  0, -2])
    """
    N = model.Nxy
    idx = np.arange(N).reshape(model.shape)
    lo = np.concatenate([idx[:-1, :].ravel(), idx[:, :-1].ravel()])
    hi = np.concatenate([idx[1:, :].ravel(), idx[:, 1:].ravel()])
    nF = len(lo)
    nFx = (model.Nx - 1) * model.Ny
    ff = np.r_[np.arange(nF), np.arange(nF)]
    ones = np.ones(nF)
    Grad = sparse.csr_matrix((np.r_[ones, -ones], (ff, np.r_[lo, hi])), shape=(nF, N))
    comp = np.r_[np.zeros(nFx, int), np.full(nF - nFx, N)]  # offset into 2nd component
    Sum = sparse.csr_matrix(
        (np.r_[ones, ones], (ff, np.r_[lo + comp, hi + comp])), shape=(nF, 2 * N)
    )
    C = model.cdarcy
    g = 2 * C * np.r_[np.full(nFx, model.hy / model.hx), np.full(nF - nFx, model.hx / model.hy)]
    return lo, hi, Grad, Sum, g


def fractional_flow(model: "ResSim", S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The water fractional flow, $ f_w = λ_w / λ_t $, and its derivative wrt. `S`."""
    Mw, Mo = model.RelPerm(S)
    dMw, dMo = model.dRelPerm(S)
    Mt = Mw + Mo
    return Mw / Mt, (dMw * Mo - Mw * dMo) / Mt**2


@dataclass
class Tape(AlignedRepr):
    """The linearization of one step of `TPFA_ResSim.ResSim.time_stepper`.

    Produced by `linearize`, consumed by `tlm_step` (and by an adjoint step,
    which needs exactly the same record). Holds the state it was taken about,
    the step's result, and the coefficients -- sparse matrices and vectors --
    of the linear statements of `tlm_step`, evaluated at that state.
    """

    __repr__ = AlignedRepr.__repr__

    model: Any
    """The model whose step this linearizes (for `RelPerm`, `ct`, ...)."""
    dt: float
    """The time step."""
    k: int
    """The time index (of the well controls)."""
    S: np.ndarray
    """Saturation at the start of the step, `(Nxy,)`."""
    P: np.ndarray
    """Pressure at the start of the step, `(Nxy,)`."""
    S1: np.ndarray
    """Saturation at the end of the step -- the recomputed forward result."""
    P1: np.ndarray
    """Pressure at the end of the step -- the recomputed forward result."""

    # Pressure equation
    Grad: Any
    """`(nF, Nxy)` face difference operator, ref `face_operators`. `Grad.T` is the divergence."""
    dMt_dS: np.ndarray
    """`(Nxy,)` derivative of the total mobility $ λ_t $ wrt. `S`."""
    dT_dMt: Any
    """`(nF, Nxy)` Jacobian of the transmissibilities wrt. the cell mobilities:
    $ ∂T_f / ∂λ_i = T_f^2 / (g_f \\, K_i \\, λ_i^2) $ for each of the face's
    two cells (harmonic averaging), `0` otherwise."""
    T: np.ndarray
    """`(nF,)` transmissibilities, $ T = g / Σ_\\mathrm{cells} 1/(K λ_t) $."""
    gradP: np.ndarray
    """`(nF,)` pressure differences, `Grad @ P1`; the fluxes are `V = T * gradP`."""
    V: np.ndarray
    """`(nF,)` fluxes through the interior faces (positive from `lo` to `hi`)."""
    accum: np.ndarray
    """`(Nxy,)` accumulation coefficient, $ φ c_t h^2 / Δt $ (`0` if `ct == 0`)."""
    solve: Callable
    """`x ↦ A⁻¹ x`, by the LU factorization of the (symmetric) pressure matrix."""

    # Well model of the BHP-controlled completions (empty if none)
    Gb: Any
    """`(nBHP, Nxy)` gathers cell values at the BHP-controlled completions; `Gb.T` scatters."""
    WI_b: np.ndarray
    """`(nBHP,)` their well indices."""
    p_bh_b: np.ndarray
    """`(nBHP,)` their bottom-hole pressures."""
    bhp_diag: np.ndarray
    """`(Nxy,)` their $ WI λ_t $, scattered onto the cells (`0` elsewhere)."""

    # Transport equation
    Q: np.ndarray
    """`(Nxy,)` total well flux per cell, signed (`realize_bhp` included)."""
    st: np.ndarray
    """`(Nxy,)` storage rate, `Q - Grad.T @ V` (`0` if `ct == 0`)."""
    dtx: np.ndarray
    """`(Nxy,)` sub-step over pore volume, `dt / nT / pv`."""
    Up: Any
    """`(nF, Nxy)` upwind selector: `(Up @ f)[face]` is `f` of the face's upwind cell."""
    Ssub: np.ndarray
    """`(nT, Nxy)` saturation at the start of each transport sub-step."""

    @property
    def nT(self) -> int:
        """Number of transport sub-steps."""
        return len(self.Ssub)


def linearize(
    model: "ResSim",
    dt: float,
    S: np.ndarray,
    P: np.ndarray | None,
    k: int = 0,
) -> Tape:
    """Recompute the step of `time_stepper` from `(S, P)` at time `k`; return its `Tape`.

    The recomputation calls the very methods of the forward model (so the
    result, `Tape.S1`/`Tape.P1`, is that of `sim`, to the solver tolerance),
    except that it records the transport sub-steps, and does not write the
    `actual_rates`/`actual_bhp` reports. Then the coefficients of the
    linearization are evaluated at that state.

    The cost is about that of a forward step plus a factorization of the
    pressure matrix (held in `Tape.solve`, for both the tangent and the
    adjoint solves).
    """
    N = model.Nxy
    S = np.asarray(S, float).ravel()
    P = np.zeros(N) if P is None else np.asarray(P, float).ravel()
    lo, hi, Grad, Sum, g = face_operators(model)
    nF = len(lo)

    # Forward step, recomputed. Mirrors `time_stepper` (minus the reporting)
    model.assemble_wells(S, P, k)
    model._validate()
    P1, VV = model.pressure_step(S, P, dt)
    model.realize_bhp(P1)
    wls = model._wells_now
    Q = model._Q  # total well flux, the BHP wells' rates now realized
    # ... and `saturation_step_upwind`, recording the sub-steps
    A_up = model.upwind_diff(VV)
    pv = model.h2 * model.por.ravel()
    fi = Q.clip(min=0)
    st = model.storage_rate(VV)
    nT = max(1, int(np.ceil(dt * model.estimate_1CFL(pv, VV, fi))))
    dtx = dt / nT / pv
    B = model._spdiags(dtx, 0) @ A_up
    Ssub = np.zeros((nT, N))
    Sj = S
    for j in range(nT):
        Ssub[j] = Sj
        Mw, Mo = model.RelPerm(Sj)
        Sj = Sj + (B @ (Mw / (Mw + Mo)) + (fi - Sj * st) * dtx)
    S1 = Sj

    # Coefficients of the linearization, at that state
    # -- mobilities and transmissibilities
    Mw, Mo = model.RelPerm(S)
    dMw, dMo = model.dRelPerm(S)
    Mt = Mw + Mo
    Kflat = model.K.reshape(-1)
    KM = model.K.reshape(2, N) * Mt  # K λ_t, both components, flattened
    KM = KM.reshape(-1)
    T = g / (Sum @ (1 / KM))
    Dup = sparse.vstack([sparse.eye(N), sparse.eye(N)])  # cell field → both components
    dT_dMt = sparse.diags(T**2 / g) @ Sum @ sparse.diags(Kflat / KM**2) @ Dup
    gradP = Grad @ P1
    V = T * gradP
    # -- the BHP wells
    is_bhp = np.isfinite(wls["WI_lam"])
    nB = int(is_bhp.sum())
    inds_b = wls["inds"][is_bhp]
    Gb = sparse.csr_matrix((np.ones(nB), (np.arange(nB), inds_b)), shape=(nB, N))
    WI = model.wells.WI
    WI_b = WI[is_bhp] if WI is not None else np.zeros(0)
    p_bh_b = wls["p_bh"][is_bhp]
    bhp_diag = wls["bhp_diag"]
    # -- the pressure system (as `TPFA` assembles it, incl. its pin)
    accum = pv * model.ct / dt if model.ct > 0 else np.zeros(N)
    diag = accum + bhp_diag
    if model.ct == 0 and not bhp_diag.any():
        diag = diag.copy()
        diag[0] += np.sum(model.K[:, 0, 0])
    A = Grad.T @ sparse.diags(T) @ Grad + sparse.diags(diag)
    solve = splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A").solve
    # -- the upwind directions
    Up = sparse.csr_matrix(
        (np.ones(nF), (np.arange(nF), np.where(V >= 0, lo, hi))), shape=(nF, N)
    )

    return Tape(
        model=model, dt=dt, k=k, S=S, P=P, S1=S1, P1=P1,
        Grad=Grad, dMt_dS=dMw + dMo, dT_dMt=dT_dMt, T=T, gradP=gradP, V=V,
        accum=accum, solve=solve,
        Gb=Gb, WI_b=WI_b, p_bh_b=p_bh_b, bhp_diag=bhp_diag,
        Q=Q, st=st, dtx=dtx, Up=Up, Ssub=Ssub,
    )  # fmt: skip


def tlm_step(tape: Tape, dS: np.ndarray, dP: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the perturbation `(dS, dP)` through the step of `tape`.

    Returns `(dS1, dP1)`. Each statement is linear in the perturbations, with
    coefficients from the tape -- ref the module docstring for how to reverse it.
    """
    t = tape
    # fmt: off
    # Pressure equation
    dMt      = t.dMt_dS * dS                                # cell mobilities λ_t
    dT       = t.dT_dMt @ dMt                               # transmissibilities
    dWI_lam  = t.WI_b * (t.Gb @ dMt)                        # BHP well model: WI λ_t
    dw       = t.Gb.T @ dWI_lam                             #   its diagonal ...
    dr       = t.Gb.T @ (t.p_bh_b * dWI_lam)                #   ... and right-hand side
    dAP      = t.Grad.T @ (t.gradP * dT) + dw * t.P1        # (δA) p¹
    dq       = t.accum * dP + dr                            # δq
    dP1      = t.solve(dq - dAP)                            # δp¹ = A⁻¹ (δq - δA p¹)
    dV       = t.T * (t.Grad @ dP1) + t.gradP * dT          # fluxes, δ(T ⊙ ∇p¹)
    dQ       = dr - dw * t.P1 - t.bhp_diag * dP1            # BHP wells' realized rates

    # Transport equation
    fp       = t.Q.clip(max=0)                              # production
    dfi      = dQ * (t.Q > 0)                               # injection
    dfp      = dQ * (t.Q < 0)
    dst      = (dQ - t.Grad.T @ dV) if t.model.ct > 0 else 0*dQ   # storage rate
    dVm      = dV * (t.V != 0)                              # d(V⁺), d(V⁻) of `upwind_diff`
    for Sj in t.Ssub:
        fw, dfw_dS = fractional_flow(t.model, Sj)
        dfw  = dfw_dS * dS
        dF   = dVm * (t.Up @ fw) + t.V * (t.Up @ dfw)       # water flux, δ(v ⊙ Up f)
        drhs = -t.Grad.T @ dF + dfp*fw + fp*dfw + dfi - dS*t.st - Sj*dst
        dS   = dS + t.dtx * drhs
    # fmt: on
    return dS, dP1


def tlm(
    model: "ResSim",
    dt: float,
    SS: np.ndarray,
    PP: np.ndarray,
    dS0: np.ndarray,
    dP0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate `(dS0, dP0)` along the trajectory `(SS, PP)` of `sim(dt, ...)`.

    Returns `(dSS, dPP)`, shaped like `SS`, `PP` (flat): the perturbed
    trajectory to first order, $ x(S_0 + ε δS_0, P_0 + ε δP_0) = x + ε δx + O(ε^2) $.
    `dP0` defaults to `0` (it is inconsequential anyway unless `ct > 0` or a
    well is BHP-controlled). Each step is re-linearized (`linearize`) on the way.
    """
    nSteps = len(SS) - 1
    dSS = np.zeros((nSteps + 1, model.Nxy))
    dPP = np.zeros((nSteps + 1, model.Nxy))
    dSS[0] = dS0
    if dP0 is not None:
        dPP[0] = dP0
    for k in range(nSteps):
        tape = linearize(model, dt, SS[k], PP[k], k)
        dSS[k + 1], dPP[k + 1] = tlm_step(tape, dSS[k], dPP[k])
    return dSS, dPP
