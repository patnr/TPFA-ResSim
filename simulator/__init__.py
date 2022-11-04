"""Main simulator code.

Implemented with OOP so as to facilitate multiple realisations, by ensuring
that the parameter values of one instance do not influence another instance.
Depending on thread-safety, this might not be necessary, but is usually cleaner
when estimating anything other than the model's input/output (i.e. the state
variables).

Note: Index ordering/labels: `x` is 1st coord., `y` is 2nd. See `grid.py` for more info.
"""
# TODO
# - Protect Nx, Ny, shape, etc?

from functools import wraps

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from struct_tools import DotDict, NicePrint
from tqdm.auto import tqdm as progbar

from simulator.grid import Grid2D


class ResSim(NicePrint, Grid2D):
    """Reservoir simulator.

    Example:
    >>> model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64)
    >>> model.config_wells(inj =[[0, .32, 1]],
    ...                    prod=[[1, 1, -1]])
    >>> water_sat0 = np.zeros(model.M)
    >>> dt = .35
    >>> nSteps = 2
    >>> S = recurse(model.time_stepper(dt), nSteps, water_sat0, pbar=False)

    This produces the following values (which are used for automatic testing):
    >>> location_inds = [100, 1300, 2900]
    >>> S[-1, location_inds]
    array([0.9429345 , 0.91358172, 0.71554613])
    """
    @wraps(Grid2D.__init__)
    def __init__(self, *args, **kwargs):

        # Init grid
        super().__init__(*args, **kwargs)

        # Gridded properties
        self.Gridded = DotDict(
            K  =np.ones((2, *self.shape)),  # permeability in x&y dirs.
            por=np.ones(self.shape),        # porosity
        )

        self.Fluid = DotDict(
            vw=1.0,   vo=1.0,  # Viscosities
            swc=0.0, sor=0.0,  # Irreducible saturations
        )

    def config_wells(self, inj, prod, remap=True):
        """Define `self.Q`, the source/sink field, i.e. injection/production wells.

        It is defined by the given list of injectors (`inj`) and producers (`prod`).
        In both lists, each entry should be a tuple: `x/Lx, y/Ly, |rate|`.

        Note:
        - The rates are scaled so as to sum to +/- 1.
          This is not stictly necessary. But it is necessary that their sum be 0,
          otherwise the model will silently input deficit from SW corner.
        - The specified well coordinates should be relative (betwen 0 and 1).
          They get co-located with grid nodes (not distributed over nearby ones).

          The well co-location does not happen if `remap` is `False`,
          which should be used in automatic, iterative optimisation,
          which changes well rates, but does not want to change anything else.
        """

        def remap_and_collocate(ww):
            """Scale rel -> abs coords. Place wells on nodes."""
            # Ensure array
            ww = np.array(ww, float)
            # Remap
            ww[:, 0] *= self.Lx
            ww[:, 1] *= self.Ly
            # Collocate
            for i in range(len(ww)):
                x, y, q = ww[i]
                ww[i, :2] = self.ind2xy(self.xy2ind(x, y))
            return ww

        if remap:
            inj  = remap_and_collocate(inj)
            prod = remap_and_collocate(prod)

        inj [:, 2] /= inj [:, 2].sum()  # noqa
        prod[:, 2] /= prod[:, 2].sum()

        # Insert in source FIELD
        Q = np.zeros(self.M)
        for x, y, q in inj:
            Q[self.xy2ind(x, y)] += q
        for x, y, q in prod:
            Q[self.xy2ind(x, y)] -= q
        assert np.isclose(Q.sum(), 0)

        self.Q = Q
        # Not used by model, but kept for reference:
        self.injectors = inj
        self.producers = prod

    # Pres() -- listing 5
    def pressure_step(self, S):
        """TPFA finite-volume of Darcy: $$ -nabla(K lambda(s) nabla(u)) = q $$."""
        # Compute K*lambda(S)
        Mw, Mo = self.RelPerm(S)
        Mt = Mw + Mo
        Mt = Mt.reshape(self.shape)
        KM = Mt * self.Gridded.K
        # Compute pressure and extract fluxes
        [P, V] = self.TPFA(KM)
        return P, V

    def spdiags(self, data, diags):
        return sparse.spdiags(data, diags, self.M, self.M)

    # RelPerm() -- listing 6
    def RelPerm(self, s, nargout_is_4=False):
        """Rel. permeabilities of oil and water."""
        Fluid = self.Fluid
        S = (s - Fluid.swc) / (1 - Fluid.swc - Fluid.sor)  # Rescale saturations
        Mw = S**2 / Fluid.vw                               # Water mobility
        Mo = (1 - S)**2 / Fluid.vo                         # Oil mobility
        if nargout_is_4:
            # Only used for implicit solver, which we don't implement
            dMw = 2 * S / Fluid.vw / (1 - Fluid.swc - Fluid.sor)
            dMo = -2 * (1 - S) / Fluid.vo / (1 - Fluid.swc - Fluid.sor)
            return Mw, Mo, dMw, dMo
        else:
            return Mw, Mo

    # TPFA() -- Listing 1
    def TPFA(self, K):
        """Two-point flux-approximation (TPFA) of Darcy:

        diffusion w/ nonlinear coefficient K.
        """
        # Compute transmissibilities by harmonic averaging.
        L = K**(-1)
        TX = np.zeros((self.Nx + 1, self.Ny))
        TY = np.zeros((self.Nx, self.Ny + 1))

        TX[1:-1, :] = 2 * self.hy / self.hx / (L[0, :-1, :] + L[0, 1:, :])
        TY[:, 1:-1] = 2 * self.hx / self.hy / (L[1, :, :-1] + L[1, :, 1:])

        # Assemble TPFA discretization matrix.
        x1 = TX[:-1, :].ravel()
        x2 = TX[1:, :] .ravel()
        y1 = TY[:, :-1].ravel()
        y2 = TY[:, 1:] .ravel()

        # Setup linear system
        DiagVecs = [-x2, -y2, y1 + y2 + x1 + x2, -y1, -x1]
        DiagIndx = [-self.Ny, -1, 0, 1, self.Ny]
        DiagVecs[2][0] += np.sum(self.Gridded.K[:, 0, 0])  # Ensure SPD ...
        # ... the DoF is thanks to working w/ a *potential*, ref article p. 13
        A = self.spdiags(DiagVecs, DiagIndx)

        # Solve
        q = self.Q
        # u = np.linalg.solve(A.A, q) # direct dense solver
        u = spsolve(A.tocsr(), q)     # direct sparse solver
        # u, _info = cg(A, q)         # conjugate gradient
        # Could also try scipy.linalg.solveh_banded which, according to
        # https://scicomp.stackexchange.com/a/30074 uses the Thomas algorithm,
        # as recommended by Aziz and Settari ("Petro. Res. simulation").
        # NB: stackexchange also mentions that solve_banded does not work well
        # when the band offsets large, i.e. higher-dimensional problems.

        # Extract fluxes
        P = u.reshape(self.shape)
        V = DotDict(
            x=np.zeros((self.Nx+1, self.Ny)),
            y=np.zeros((self.Nx, self.Ny+1)),
        )
        V.x[1:-1,:] = (P[:-1,:] - P[1:,:]) * TX[1:-1,:]  # noqa
        V.y[:,1:-1] = (P[:,:-1] - P[:,1:]) * TY[:,1:-1]  # noqa
        return P, V

    # GenA() -- listing 7
    def upwind_diff(self, V):
        """Upwind finite-volume scheme."""
        fp = self.Q.clip(max=0)  # production
        # Flow fluxes, separated into direction (x-y) and sign
        x1 = V.x.clip(max=0)[:-1, :].ravel()
        y1 = V.y.clip(max=0)[:, :-1].ravel()
        x2 = V.x.clip(min=0)[1:, :] .ravel()
        y2 = V.y.clip(min=0)[:, 1:] .ravel()
        DiagVecs = [x2, y2, fp + y1 - y2 + x1 - x2, -y1, -x1]
        DiagIndx = [-self.Ny, -1, 0, 1, self.Ny]
        A = self.spdiags(DiagVecs, DiagIndx)
        return A

    # Extracted from Upstream()
    def estimate_CFL(self, pv, V, fi):
        """Estimate CFL for use with saturation_step_upwind()."""
        # In-/Out-flux x-/y- faces
        XP = V.x.clip(min=0)
        XN = V.x.clip(max=0)
        YP = V.y.clip(min=0)
        YN = V.y.clip(max=0)
        Vi = XP[:-1, :] + YP[:, :-1] - XN[1:, :] - YN[:, 1:]

        pm  = min(pv / (Vi.ravel() + fi))  # estimate of influx
        sat = self.Fluid.swc + self.Fluid.sor
        cfl = ((1 - sat) / 3) * pm  # NB: 3-->2 since no z-dim ?
        return cfl

    # Upstream() -- listing 8
    def saturation_step_upwind(self, S, V, dt):
        """Explicit upwind finite-volume discretisation of CoM."""
        # Compute dt
        pv = self.h2 * self.Gridded.por.ravel()  # Pore volume = cell volume * porosity
        fi = self.Q.clip(min=0)                  # Well inflow
        cfl = self.estimate_CFL(pv, V, fi)
        Nts = int(np.ceil(dt / cfl))             # num. (local) time steps
        dtx = (dt / Nts) / pv                    # (local) time steps

        # Discretized transport operator
        A = self.upwind_diff(V)                  # system matrix
        A = self.spdiags(dtx, 0) @ A             # A * dt/|Omega i|

        for _ in range(Nts):
            mw, mo = self.RelPerm(S)             # compute mobilities
            fw = mw / (mw + mo)                  # compute fractional flow
            S = S + (A @ fw + fi * dtx)          # update saturation
        return S

    def time_stepper(self, dt):
        def integrate(S):
            [P, V] = self.pressure_step(S)
            S      = self.saturation_step_upwind(S, V, dt)
            return S
        return integrate


def recurse(fun, nSteps, x0, pbar=True):
    """Recursively apply `fun` `nSteps` times.

    Note: `output[0] == x0`, hence `len(output) = nSteps + 1`.

    BTW, "recurse" is a fancy programming term referring to a function calling itself.
    Here we implement it simply by a for loop, passing previous output as next intput.
    Indeed "recursive" is also an accurate description of causal (Markov) processes,
    such as nature or its simulators, which build on themselves.
    """
    # Init
    xx = np.zeros((nSteps+1,)+x0.shape)
    xx[0] = x0

    # Recurse
    kk = np.arange(nSteps)
    for k in (progbar(kk, "Simulation") if pbar else kk):
        xx[k+1] = fun(xx[k])
    return xx


if __name__ == "__main__":
    pass
