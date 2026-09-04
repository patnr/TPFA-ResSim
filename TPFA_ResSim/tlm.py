"""The tangent linear model (TLM) of `TPFA_ResSim.ResSim.time_stepper`, and its adjoint, by hand.

I.e. the Jacobian of one time step, $ (s^n, p^n) ↦ (s^{n+1}, p^{n+1}) $, with
respect to the *state* -- and to the parameter $ \\log K $ -- applied to a
perturbation (`tlm_step`) or transposed onto a sensitivity (`adj_step`), rather
than formed (it is dense, through $A^{-1}$). Chained along a trajectory
(`tlm`, `adjoint`), the latter yields the gradient of an objective with respect
to the initial state, `S0` and `P0` of `TPFA_ResSim.ResSim.sim`, and to
$ \\log K $, at the cost of one more sweep -- whatever the number of parameters.

`linearize` recomputes one forward step -- from the state it is given, so the
trajectory that `sim` returns is all the record it needs (checkpointing) -- and
returns a `Tape` of the intermediates. `tlm_step` then propagates a perturbation
through that step, and `adj_step` a sensitivity back through it. `tlm` does the
former along a whole trajectory, which the following checks against a finite
difference:

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
>>> dlogK = rng.standard_normal(model.K.shape)
>>> dSS, dPP = tlm(model, dt, SS, PP, dS0, dP0, dlogK)

>>> eps, K = 1e-6, model.K.copy()
>>> model.K = K * np.exp(+eps*dlogK)
>>> SSp, PPp = model.sim(dt, nSteps, S0 + eps*dS0, P0 + eps*dP0, pbar=False)
>>> model.K = K * np.exp(-eps*dlogK)
>>> SSm, PPm = model.sim(dt, nSteps, S0 - eps*dS0, P0 - eps*dP0, pbar=False)
>>> model.K = K
>>> bool(np.abs((SSp - SSm) / (2*eps) - dSS).max() < 1e-6)
True
>>> bool(np.abs((PPp - PPm) / (2*eps) - dPP).max() < 1e-6)
True

(`cached_precond=False` merely spares the finite difference the noise of the
iterative solver's tolerance, $10^{-10}$, which `eps` would amplify a
million-fold; the TLM itself does not care.)

## Seeding the adjoint with an objective

`adjoint` sweeps backwards along the trajectory, from the partial derivatives
of a scalar objective, $ J(S, P) $, with respect to the *stored* states --
`dJ_dSS[k]` $ = ∂J/∂S_k $ and `dJ_dPP[k]` $ = ∂J/∂P_k $, shaped like `SS` and
`PP` -- and returns $ ∂J/∂S_0 $, $ ∂J/∂P_0 $ and $ ∂J/∂\\log K $ (a `Gradient`).
The seeds are simply what the objective says they are:

- A quantity of the *final* state seeds index `-1` alone. E.g. the water
  saturation at the producer, $ J = S_N[i_\\mathrm{prd}] $:

  >>> prd = model.xy2ind(1, 1)
  >>> dJ_dSS = np.zeros_like(SS)
  >>> dJ_dSS[-1, prd] = 1
  >>> grad = adjoint(model, dt, SS, PP, dJ_dSS)

  whose directional derivative along any $ (δS_0, δP_0, δ\\log K) $ is what
  the TLM propagates:

  >>> lhs = dSS[-1, prd]
  >>> rhs = grad.S0 @ dS0 + grad.P0 @ dP0 + (grad.logK * dlogK).sum()
  >>> bool(abs(lhs - rhs) < 1e-10 * abs(lhs))
  True

- A data misfit, $ J = \\frac{1}{2} \\sum_k \\| H_k S_k - y_k \\|^2 / σ^2 $ (the
  history-matching case), seeds every observed time with its weighted
  residual, scattered back onto the observed cells:
  `dJ_dSS[k] = H_k.T @ (H_k @ SS[k] - y_k) / σ**2`.
- Objectives on the well *reports* (`actual_rates`, `actual_bhp`) are not
  seedable directly: those are functions of $ (S_k, P_{k+1}) $ through the
  well model (`TPFA_ResSim.ResSim.bhp`, `TPFA_ResSim.ResSim.realize_bhp`),
  which must be differentiated by hand into the seeds. Not built in.

The gradient with respect to $ \\log K $ has the shape of `K`, `(2, Nx, Ny)`:
both permeability components. For an *isotropic* field (a scalar or
`(Nx, Ny)`-shaped `K`, which `ResSim.__setattr__` broadcasts to both), the
gradient with respect to the single $ \\log k $ field is `grad.logK.sum(0)`.

## Written to be reversed

`tlm_step` is a straight-line sequence of statements (the one loop being over
the transport sub-steps), each linear in the perturbations, and each of one of
three kinds:

- `y = M @ x`, with `M` a sparse matrix held by the tape (`Tape.Grad`,
  `Tape.Up`, ...);
- `y = a * x`, with `a` a vector (i.e. a diagonal matrix) held by the tape,
  or computed from it;
- `y = tape.solve(x)`, the pressure solve, $ A^{-1} x $.

`adj_step` is the same statements in *reverse* order, each transposed --
`x̄ += M.T @ ȳ`, `x̄ += a * ȳ` -- with the pressure solve its own transpose,
$ A $ being symmetric (that is what makes the TPFA system SPD, ref
`TPFA_ResSim.ResSim.cached_precond`), so `Tape.solve` serves both directions
unchanged. Nothing else happens in there: no indexing gymnastics (the
5-diagonal assemblies of `TPFA_ResSim.ResSim.TPFA` and
`TPFA_ResSim.ResSim.upwind_diff` are recast on the face operators below, whose
transposes are just `.T`), no branching on the perturbations. The two are
verified against each other by the dot-product test of `tests/test_tlm.py`,
$ ⟨ \\mathbf{M} δx, \\bar{y} ⟩ = ⟨ δx, \\mathbf{M}^T \\bar{y} ⟩ $, to round-off.

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
$λ_t(s^n)$ and of $K$, ref `Tape.dT_dMt`, `Tape.dT_dlogK`), $a$ the
accumulation term (`Tape.accum`) and $w$, $r$ the well model of the
BHP-controlled wells (`Tape.bhp_diag`, and the matching right-hand side, both
proportional to $λ_t$ in the well cells); then, for each of the `nT` sub-steps,

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
- The **permeability**, as $ \\log K $ (positivity built in, and the natural
  history-matching parameter), through the transmissibilities alone. Its two
  other appearances are *not* differentiated, and rightly so: the pin of the
  incompressible pressure system (`TPFA` adds $ \\sum K_{00} $ to the first
  diagonal entry) fixes $ p_0 = 0 $ whatever the value, so the derivative is
  exactly zero; and the well index `Wells.WI` is a *stored* parameter
  (`TPFA_ResSim.wells.peaceman_WI` evaluates it once, from the `K` of that
  moment, and does not track `K` thereafter), so it is held fixed, like the
  other well parameters.
- **Not** the other parameters: `por`, `ct`, the viscosities, the well
  positions and specifications. These stay fixed, as they are in `sim`.
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

.. warning:: `linearize` mutates the model, as a forward step does.

    Ref its docstring. In short: it leaves `_Q`, `_wells_now` and the
    preconditioner cache `_pLU` as of the step it linearized -- after a
    reverse sweep, those of the *first* step -- but not the reports.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from TPFA_ResSim._repr import AlignedRepr

if TYPE_CHECKING:
    from TPFA_ResSim import ResSim


class Gradient(NamedTuple):
    """The gradient of an objective, as returned by `adjoint`."""

    S0: np.ndarray
    """W.r.t. the initial saturation, `(Nxy,)`."""
    P0: np.ndarray
    """W.r.t. the initial pressure, `(Nxy,)`. Zero unless `ct > 0` or a well is on BHP."""
    logK: np.ndarray
    """W.r.t. $ \\log K $, shaped like `K`: `(2, Nx, Ny)`. Sum over axis `0` if isotropic."""


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

    Produced by `linearize`, consumed by `tlm_step` and `adj_step`. Holds the
    state it was taken about, the step's result, and the coefficients -- sparse
    matrices and vectors -- of the linear statements of `tlm_step`, evaluated
    at that state.
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
    dT_dlogK: Any
    """`(nF, 2*Nxy)` Jacobian of the transmissibilities wrt. $ \\log K $ (flattened
    like `K`): $ ∂T_f / ∂\\log K_c = T_f^2 / (g_f \\, K_c \\, λ_c) $ for each of the
    face's two cells, in the face's direction, `0` otherwise."""
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

    .. warning:: This mutates the model, exactly as a step of `sim` does.

        Because it *is* one: `assemble_wells` (hence `well_controls`),
        `pressure_step` and `realize_bhp` are called, and they write the
        source field `_Q`, the bundle `_wells_now` and -- if `cached_precond`
        -- the factorization cache `_pLU`, all as of the step linearized. So
        after `sim`, then `adjoint` (which linearizes the steps in *reverse*),
        these hold the values of the first step rather than the last. None of
        it is consequential: the next step (or `linearize`) overwrites them,
        and `_pLU` is a mere preconditioner (ref `_solve_pressure`). The
        well reports, `actual_rates`/`actual_bhp`, are *not* written, so they
        remain those of the `sim` that produced the trajectory. Nothing else
        is touched: `K`, `por`, the well specifications are read only.
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
    dT_dKM = sparse.diags(T**2 / g) @ Sum  # ... @ diag(1/KM²), below
    dT_dMt = dT_dKM @ sparse.diags(Kflat / KM**2) @ Dup
    dT_dlogK = dT_dKM @ sparse.diags(1 / KM)  # since ∂(Kλ)/∂log K = Kλ
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
        Grad=Grad, dMt_dS=dMw + dMo, dT_dMt=dT_dMt, dT_dlogK=dT_dlogK,
        T=T, gradP=gradP, V=V, accum=accum, solve=solve,
        Gb=Gb, WI_b=WI_b, p_bh_b=p_bh_b, bhp_diag=bhp_diag,
        Q=Q, st=st, dtx=dtx, Up=Up, Ssub=Ssub,
    )  # fmt: skip


def tlm_step(
    tape: Tape,
    dS: np.ndarray,
    dP: np.ndarray,
    dlogK: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the perturbation `(dS, dP, dlogK)` through the step of `tape`.

    Returns `(dS1, dP1)`. `dlogK` is shaped like `K` (or `None`: `0`). Each
    statement is linear in the perturbations, with coefficients from the tape;
    `adj_step` is its reverse, statement by statement.
    """
    t = tape
    dlogK = np.zeros(2 * len(dS)) if dlogK is None else np.asarray(dlogK).reshape(-1)
    # fmt: off
    # Pressure equation
    dMt      = t.dMt_dS * dS                                # cell mobilities λ_t
    dT       = t.dT_dMt @ dMt + t.dT_dlogK @ dlogK          # transmissibilities
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


def adj_step(
    tape: Tape, aS1: np.ndarray, aP1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate the sensitivity `(aS1, aP1)` back through the step of `tape`.

    The transpose of `tlm_step`: given $ ∂J/∂s^{n+1} $ and $ ∂J/∂p^{n+1} $,
    returns `(aS, aP, alogK)` $ = (∂J/∂s^n, ∂J/∂p^n, ∂J/∂\\log K) $, the last
    shaped like `K` and being this step's *contribution* (to be summed over
    the steps, as `adjoint` does). Each statement of `tlm_step` appears here,
    in reverse order, transposed: `y = M @ x` as `x̄ += M.T @ ȳ`, `y = a * x`
    as `x̄ += a * ȳ`, the (symmetric) solve as itself. The comments name the
    statement being transposed.

    Does not modify `aS1`, `aP1`.
    """
    t = tape
    aS = np.array(aS1, float)  # the running adjoint of `dS`, which the loop updates
    aP1 = np.array(aP1, float)  # `dP1` is the output, and feeds `dV` and `dQ`
    zeros = np.zeros(len(aS))
    adfi, adfp, adst, adVm = zeros, zeros, zeros, np.zeros_like(t.V)
    fp = t.Q.clip(max=0)
    # fmt: off
    # Transport equation, sub-steps in reverse
    for Sj in t.Ssub[::-1]:
        fw, dfw_dS = fractional_flow(t.model, Sj)
        adrhs = t.dtx * aS                                  # dS = dS + dtx*drhs
        adF   = -(t.Grad @ adrhs)                           # drhs = -Grad.T @ dF ...
        adfp  = adfp + fw * adrhs                           #   + dfp*fw
        adfw  = fp * adrhs                                  #   + fp*dfw
        adfi  = adfi + adrhs                                #   + dfi
        aS    = aS - t.st * adrhs                           #   - dS*st
        adst  = adst - Sj * adrhs                           #   - Sj*dst
        adVm  = adVm + (t.Up @ fw) * adF                    # dF = dVm*(Up@fw) ...
        adfw  = adfw + t.Up.T @ (t.V * adF)                 #   + V*(Up@dfw)
        aS    = aS + dfw_dS * adfw                          # dfw = dfw_dS * dS
    adV  = (t.V != 0) * adVm                                # dVm = dV * (V != 0)
    if t.model.ct > 0:                                      # dst = dQ - Grad.T @ dV
        adQ = adst
        adV = adV - t.Grad @ adst
    else:
        adQ = zeros
    adQ  = adQ + (t.Q < 0) * adfp + (t.Q > 0) * adfi        # dfp, dfi = dQ*(Q<0), dQ*(Q>0)

    # Pressure equation
    adr  = adQ                                              # dQ = dr - dw*P1 - bhp_diag*dP1
    adw  = -t.P1 * adQ
    aP1  = aP1 - t.bhp_diag * adQ
    aP1  = aP1 + t.Grad.T @ (t.T * adV)                     # dV = T*(Grad@dP1) + gradP*dT
    adT  = t.gradP * adV
    adq  = t.solve(aP1)                                     # dP1 = solve(dq - dAP)
    adAP = -adq
    aP   = t.accum * adq                                    # dq = accum*dP + dr
    adr  = adr + adq
    adT  = adT + t.gradP * (t.Grad @ adAP)                  # dAP = Grad.T@(gradP*dT) + dw*P1
    adw  = adw + t.P1 * adAP
    adWI_lam = t.p_bh_b * (t.Gb @ adr)                      # dr = Gb.T @ (p_bh_b * dWI_lam)
    adWI_lam = adWI_lam + t.Gb @ adw                        # dw = Gb.T @ dWI_lam
    adMt  = t.Gb.T @ (t.WI_b * adWI_lam)                    # dWI_lam = WI_b * (Gb @ dMt)
    adMt  = adMt + t.dT_dMt.T @ adT                         # dT = dT_dMt@dMt + dT_dlogK@dlogK
    alogK = t.dT_dlogK.T @ adT
    aS    = aS + t.dMt_dS * adMt                            # dMt = dMt_dS * dS
    # fmt: on
    return aS, aP, alogK.reshape(t.model.K.shape)


def tlm(
    model: "ResSim",
    dt: float,
    SS: np.ndarray,
    PP: np.ndarray,
    dS0: np.ndarray,
    dP0: np.ndarray | None = None,
    dlogK: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate `(dS0, dP0, dlogK)` along the trajectory `(SS, PP)` of `sim(dt, ...)`.

    Returns `(dSS, dPP)`, shaped like `SS`, `PP` (flat): the perturbed
    trajectory to first order,
    $ x(S_0 + ε δS_0, P_0 + ε δP_0, K e^{ε δ\\log K}) = x + ε δx + O(ε^2) $.
    `dP0` and `dlogK` default to `0` (the former is inconsequential anyway
    unless `ct > 0` or a well is BHP-controlled). Each step is re-linearized
    (`linearize`) on the way.
    """
    nSteps = len(SS) - 1
    dSS = np.zeros((nSteps + 1, model.Nxy))
    dPP = np.zeros((nSteps + 1, model.Nxy))
    dSS[0] = dS0
    if dP0 is not None:
        dPP[0] = dP0
    for k in range(nSteps):
        tape = linearize(model, dt, SS[k], PP[k], k)
        dSS[k + 1], dPP[k + 1] = tlm_step(tape, dSS[k], dPP[k], dlogK)
    return dSS, dPP


def adjoint(
    model: "ResSim",
    dt: float,
    SS: np.ndarray,
    PP: np.ndarray,
    dJ_dSS: np.ndarray,
    dJ_dPP: np.ndarray | None = None,
) -> Gradient:
    """The gradient of $ J(S, P) $ wrt. `S0`, `P0` and $ \\log K $, by the adjoint sweep.

    Seeded by the partials of the objective wrt. the *stored* trajectory,
    `dJ_dSS[k]` $ = ∂J/∂S_k $, `dJ_dPP[k]` $ = ∂J/∂P_k $ (shaped like `SS`,
    `PP`; the latter defaults to `0`), ref the module docstring. Sweeps
    backwards from the final time, re-linearizing each step (`linearize`) on
    the way, so the trajectory `(SS, PP)` of `sim(dt, ...)` is all it needs.

    The cost is about that of a `sim` (one forward step and one factorization
    per step, ref `linearize`), independently of the number of parameters --
    which is the point of an adjoint.
    """
    nSteps = len(SS) - 1
    aS = np.array(dJ_dSS[-1], float)
    aP = np.zeros(model.Nxy) if dJ_dPP is None else np.array(dJ_dPP[-1], float)
    alogK = np.zeros(model.K.shape)
    for k in reversed(range(nSteps)):
        tape = linearize(model, dt, SS[k], PP[k], k)
        aS, aP, aK = adj_step(tape, aS, aP)
        alogK += aK
        aS += dJ_dSS[k]
        if dJ_dPP is not None:
            aP += dJ_dPP[k]
    return Gradient(aS, aP, alogK)
