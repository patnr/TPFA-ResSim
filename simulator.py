"""From [ref](http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf)"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from struct_tools import DotDict


# Pres() -- listing 5
def pressure_step(Grid, S, Fluid, q):
    """TPFA finite-volume of Darcy: $$ -nabla(K lambda(s) nabla(u)) = q $$."""
    # Compute K*lambda(S)
    Mw, Mo = RelPerm(S, Fluid)
    Mt = Mw + Mo
    Mt = Mt.reshape((Grid.Nx, Grid.Ny))
    KM = Mt * Grid.K
    # Compute pressure and extract fluxes
    [P, V] = TPFA(Grid, KM, q)
    return P, V


# RelPerm() -- listing 6
def RelPerm(s, Fluid, nargout_is_4=False):
    """Rel. permeabilities of oil and water."""
    S = (s - Fluid.swc) / (1 - Fluid.swc - Fluid.sor)  # Rescale saturations
    Mw = S**2 / Fluid.vw                               # Water mobility
    Mo = (1 - S)**2 / Fluid.vo                         # Oil mobility
    if nargout_is_4:
        dMw = 2 * S / Fluid.vw / (1 - Fluid.swc - Fluid.sor)
        dMo = -2 * (1 - S) / Fluid.vo / (1 - Fluid.swc - Fluid.sor)
        return Mw, Mo, dMw, dMo
    else:
        return Mw, Mo


# TPFA() -- Listing 1
def TPFA(Grid,K,q):
    """Two-point flux-approximation (TPFA) of Darcy:

    diffusion w/ nonlinear coefficient K.
    """
    # Compute transmissibilities by harmonic averaging.
    L = K**(-1)
    TX = np.zeros((Grid.Nx + 1, Grid.Ny))
    TY = np.zeros((Grid.Nx, Grid.Ny + 1))

    TX[1:-1, :] = 2 * Grid.hy / Grid.hx / (L[0, :-1, :] + L[0, 1:, :])
    TY[:, 1:-1] = 2 * Grid.hx / Grid.hy / (L[1, :, :-1] + L[1, :, 1:])

    # Assemble TPFA discretization matrix.
    x1 = TX[:-1, :].ravel()
    x2 = TX[1:, :] .ravel()
    y1 = TY[:, :-1].ravel()
    y2 = TY[:, 1:] .ravel()

    # Setup linear system
    DiagVecs = [-x2, -y2, y1 + y2 + x1 + x2, -y1, -x1]
    DiagIndx = [-Grid.Ny, -1, 0, 1, Grid.Ny]
    DiagVecs[2][0] += np.sum(Grid.K[:, 0, 0])  # Ensure SPD ...
    # ... the DoF is thanks to working w/ a *potential*, ref article p. 13
    A = sparse.spdiags(DiagVecs, DiagIndx, Grid.N, Grid.N)

    # Solve
    # u = np.linalg.solve(A.A, q) # direct dense solver
    u = spsolve(A.tocsr(), q)     # direct sparse solver
    # u, _info = cg(A, q)         # conjugate gradient
    # Could also try scipy.linalg.solveh_banded which, according to
    # https://scicomp.stackexchange.com/a/30074 uses the Thomas algorithm,
    # as recommended by Aziz and Settari ("Petro. Res. simulation").
    # NB: stackexchange also mentions that solve_banded does not work well
    # when the band offsets large, i.e. higher-dimensional problems.

    # Extract fluxes
    P = u.reshape((Grid.Nx, Grid.Ny))
    V = DotDict(
        x=np.zeros((Grid.Nx+1, Grid.Ny)),
        y=np.zeros((Grid.Nx, Grid.Ny+1)),
    )
    V.x[1:-1,:] = (P[:-1,:] - P[1:,:]) * TX[1:-1,:]
    V.y[:,1:-1] = (P[:,:-1] - P[:,1:]) * TY[:,1:-1]
    return P, V


# GenA() -- listing 7
def upwind_diff(Grid, V, q):
    """Upwind finite-volume scheme."""
    fp = q.clip(max=0)  # production
    # Flow fluxes, separated into direction (x-y) and sign
    x1 = V.x.clip(max=0)[:-1, :].ravel()
    y1 = V.y.clip(max=0)[:, :-1].ravel()
    x2 = V.x.clip(min=0)[1:, :] .ravel()
    y2 = V.y.clip(min=0)[:, 1:] .ravel()
    DiagVecs = [x2, y2, fp + y1 - y2 + x1 - x2, -y1, -x1]
    DiagIndx = [-Grid.Ny, -1, 0, 1, Grid.Ny]
    A = sparse.spdiags(DiagVecs, DiagIndx, Grid.N, Grid.N)
    return A


# Extracted from Upstream()
def estimate_CFL(pv, Fluid, V, fi):
    """Estimate CFL for use with saturation_step_upwind()."""
    # In-/Out-flux x-/y- faces
    XP = V.x.clip(min=0)
    XN = V.x.clip(max=0)
    YP = V.y.clip(min=0)
    YN = V.y.clip(max=0)
    Vi = XP[:-1, :] + YP[:, :-1] - XN[1:, :] - YN[:, 1:]

    pm  = min(pv / (Vi.ravel() + fi))  # estimate of influx
    sat = Fluid.swc + Fluid.sor
    cfl = ((1 - sat) / 3) * pm  # NB: 3-->2 since no z-dim ?
    return cfl


# Upstream() -- listing 8
def saturation_step_upwind(Grid, S, Fluid, V, q, T):
    """Explicit upwind finite-volume discretisation of CoM."""
    # Compute dt
    pv = Grid.h2 * Grid.por.ravel()  # Pore volume = cell volume * porosity
    fi = q.clip(min=0)               # Well inflow
    cfl = estimate_CFL(pv, Fluid, V, fi)
    Nts = int(np.ceil(T / cfl))      # num. (local) time steps
    dtx = (T / Nts) / pv             # (local) time steps

    # Discretized transport operator
    A = upwind_diff(Grid, V, q)                     # system matrix
    A = sparse.spdiags(dtx, 0, Grid.N, Grid.N) @ A  # A * dt/|Omega i|

    for _ in range(Nts):
        mw, mo = RelPerm(S, Fluid)   # compute mobilities
        fw = mw / (mw + mo)          # compute fractional flow
        S = S + (A @ fw + fi * dtx)  # update saturation
    return S


if __name__ == "__main__":
    # Grid settings
    Grid = DotDict(
        Dx=1,
        Dy=1,
        Nx=64,
        Ny=64,
    )
    Grid.N = Grid.Nx * Grid.Ny

    # Cell dims
    Grid.hx  = Grid.Dx / Grid.Nx
    Grid.hy  = Grid.Dy / Grid.Ny
    Grid.h2  = Grid.hx * Grid.hy

    Grid.K   = np.ones((2, Grid.Nx, Grid.Ny)) # Unit permeability
    Grid.por = np.ones((Grid.Nx, Grid.Ny))    # Unit porosity

    # Source terms: production/injection
    Q     = np.zeros(Grid.N)
    Q[20] = +1
    Q[-1] = -1

    Fluid = DotDict(
        vw=1.0, vo=1.0,    # Viscosities
        swc=0.0, sor=0.0,  # Irreducible saturations
    )

    # nSteps=28 used in paper
    nSteps = 2
    S = np.zeros(Grid.N)  # Initial saturation

    dt = 0.7 / nSteps
    for _ in range(nSteps):
        [P, V] = pressure_step(Grid, S, Fluid, Q)
        S = saturation_step_upwind(Grid, S, Fluid, V, Q, dt)

    # I have cross-checked the output of this code with that of the Matlab code,
    # and ensured that they produce the same values. Example locations/values:
    assert np.isclose(S[100] , 0.9429344998048418)
    assert np.isclose(S[1300], 0.9135817175788589)
    assert np.isclose(S[2900], 0.7155461308680394)
