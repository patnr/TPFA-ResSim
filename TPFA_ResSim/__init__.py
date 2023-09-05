""".. include:: README.md"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from struct_tools import DotDict, NicePrint
from tqdm.auto import tqdm

from TPFA_ResSim.grid import Grid2D
from TPFA_ResSim.plotting import Plot2D


class ResSim(NicePrint, Grid2D, Plot2D):
    """Reservoir simulator class.

    Implemented with OOP (instead of passing around dicts) to facilitate
    bookkeeping of ensemble forecasting
    (where parameter values of one instance should not influence another)

    Example:
    >>> model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64)
    >>> model.inj_xy=[[0, .32]]
    >>> model.inj_rates=[[1]]
    >>> model.prod_rates=[[1]]
    >>> model.prod_xy=[[1, 1]]
    >>> water_sat0 = np.zeros(model.Nxy)
    >>> dt = .35
    >>> nSteps = 2
    >>> S = recurse(model.time_stepper(dt), nSteps, water_sat0, pbar=False)

    This produces the following values (used for automatic testing):
    >>> S[-1, [100, 1300, 2900]]
    array([0.9429345 , 0.91358172, 0.71554613])
    """
    def __init__(self, **kwargs):
        """Write keyword arguments to `self`."""

        # Init super
        super().__init__(**{k: kwargs.pop(k) for k in list(kwargs)
                            if k in ["Lx", "Ly", "Nx", "Ny"]})

        # Add defaults
        defaults = dict(K=np.ones((2, *self.shape)), por=np.ones(self.shape),
                        vw=1., vo=1., swc=0., sor=0.)
        kwargs = dict(defaults, **kwargs)

        # Write
        for k in list(kwargs):
            setattr(self, k, kwargs.pop(k))


    K: np.ndarray
    """Permeabilities (in x and y directions). Array of shape `(2, Nx, Ny)`)."""
    por: np.ndarray
    """Porosity; Array of shape `(Nx, Ny)`)."""
    vw: float
    """Viscosity for water. Defaults: 1."""
    vo: float
    """Viscosity for oil. Defaults: 1."""
    swc: float
    """Irreducible saturation, water. Default: 0"""
    sor: float
    """Irreducible saturation, oil. Default: 0"""

    @property
    def inj_xy(self):
        """Array of shape `(n, 2)` of x- and y-coords for `n` injector wells.

        Values should be betwen `0` and `Lx` or `Ly`.

        .. warning:: The wells get co-located with grid nodes, not distributed over nearby ones.
            This is a design choice, not a mathematical necessity.
        """
        return self._inj_xy

    @property
    def prod_xy(self):
        """Like `inj_xy`, but for producing wells."""
        return self._prod_xy

    def _collocate_at_node(self, xys):
        xys = np.array(xys, float)
        for i, (x, y) in enumerate(xys):
            xys[i] = self.ind2xy(self.xy2ind(x, y))
        return xys

    @inj_xy.setter
    def inj_xy(self, value):
        self._inj_xy = self._collocate_at_node(value)

    @prod_xy.setter
    def prod_xy(self, value):
        self._prod_xy = self._collocate_at_node(value)

    @property
    def inj_rates(self):
        """Array of shape `(n, nTime)` -- or `(n, 1)` if constant-in-time.

        .. note:: Both `inj_rates` and `prod_rates` are rates should be positive.
            At each time index, it is asserted that the rates sum to 0,
            otherwise the model would silently input deficit from SW corner.
        """
        return self._inj_rates
    @property
    def prod_rates(self):
        """Like `prod_rates`, but for producing wells."""
        return self._prod_rates

    @inj_rates.setter
    def inj_rates(self, value):
        self._inj_rates = np.array(value, float)

    @prod_rates.setter
    def prod_rates(self, value):
        self._prod_rates = np.array(value, float)


    def _set_Q(self, k):
        """Define source/sink *field* (at time `k`) from well specifications."""
        Q = np.zeros(self.Nxy)
        for xys, sign, rates in [(self.inj_xy, +1, self.inj_rates),
                                 (self.prod_xy, -1, self.prod_rates)]:

            if rates.shape[1] == 1:
                rates = rates[:, 0]
            else:
                rates = rates[:, k]
            assert len(rates) == len(xys), "Mismatch in rate specification"

            for xy, q in zip(xys, rates):
                # Use += in case of superimposed wells (e.g. by optimzt)
                Q[self.xy2ind(*xy)] += sign * q
        assert np.isclose(Q.sum(), 0), "Inj - Prod does not sum to 0"
        self._Q = Q

    # Pres() -- listing 5
    def pressure_step(self, S):
        """Compute permeabilities then solve Darcy's equation. Returns `[P, V]`."""
        # Compute K*λ(S)
        Mw, Mo = self.RelPerm(S)
        Mt = Mw + Mo
        Mt = Mt.reshape(self.shape)
        KM = Mt * self.K
        # Compute pressure and extract fluxes
        [P, V] = self.TPFA(KM)
        return P, V

    def _spdiags(self, data, diags):
        return sparse.spdiags(data, diags, self.Nxy, self.Nxy)

    def rescale_sat(self, s):
        """Account for irreducible saturations. Ref paper, p. 32."""
        return (s - self.swc) / (1 - self.swc - self.sor)

    # RelPerm() -- listing 6
    def RelPerm(self, s):
        """Rel. permeabilities of oil and water. Return as mobilities (perm/viscocity)."""
        S = self.rescale_sat(s)
        Mw = S**2 / self.vw        # Water mobility
        Mo = (1 - S)**2 / self.vo  # Oil mobility
        return Mw, Mo

    def dRelPerm(self, s):
        """Derivatives of `RelPerm`."""
        S = self.rescale_sat(s)
        dMw = 2 * S / self.vw / (1 - self.swc - self.sor)
        dMo = -2 * (1 - S) / self.vo / (1 - self.swc - self.sor)
        return dMw, dMo

    # TPFA() -- Listing 1
    def TPFA(self, K):
        """Two-point flux-approximation (TPFA) of Darcy: $ -∇(K ∇u) = q $

        i.e. steady-state diffusion w/ nonlinear coefficient, $K$.

        After solving for pressure `P`, extract the fluxes `V`
        by finite differences.
        """
        # Compute transmissibilities by harmonic averaging.
        L = 1/K
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
        DiagVecs[2][0] += np.sum(self.K[:, 0, 0])  # ref article p. 13
        A = self._spdiags(DiagVecs, DiagIndx)

        # Solve; compute A\q
        q = self._Q
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
        V.x[1:-1, :] = (P[:-1, :] - P[1:, :]) * TX[1:-1, :]
        V.y[:, 1:-1] = (P[:, :-1] - P[:, 1:]) * TY[:, 1:-1]
        return P, V

    # GenA() -- listing 7
    def upwind_diff(self, V):
        """Upwind finite-volume scheme."""
        fp = self._Q.clip(max=0)  # production
        # Flow fluxes, separated into direction (x-y) and sign
        x1 = V.x.clip(max=0)[:-1, :].ravel()
        y1 = V.y.clip(max=0)[:, :-1].ravel()
        x2 = V.x.clip(min=0)[1:, :] .ravel()
        y2 = V.y.clip(min=0)[:, 1:] .ravel()
        DiagVecs = [x2, y2, fp + y1 - y2 + x1 - x2, -y1, -x1]
        DiagIndx = [-self.Ny, -1, 0, 1, self.Ny]
        A = self._spdiags(DiagVecs, DiagIndx)
        return A

    # Extracted from Upstream()
    def estimate_CFL(self, pv, V, fi):
        """Estimate CFL for use with `saturation_step_upwind`."""
        # In-/Out-flux x-/y- faces
        XP = V.x.clip(min=0)
        XN = V.x.clip(max=0)
        YP = V.y.clip(min=0)
        YN = V.y.clip(max=0)
        Vi = XP[:-1, :] + YP[:, :-1] - XN[1:, :] - YN[:, 1:]

        pm  = min(pv / (Vi.ravel() + fi))  # estimate of influx
        sat = self.swc + self.sor
        cfl = ((1 - sat) / 3) * pm  # NB: 3-->2 since no z-dim ?
        return cfl

    # Upstream() -- listing 8
    def saturation_step_upwind(self, S, V, dt):
        """Explicit upwind FV discretisation of conserv. of mass (water sat.)."""
        A  = self.upwind_diff(V)                 # FV discretized transport operator
        pv = self.h2 * self.por.ravel()          # Pore volume = cell volume * porosity
        fi = self._Q.clip(min=0)                 # Well inflow

        # Compute sub/local dt
        cfl = self.estimate_CFL(pv, V, fi)
        nT = int(np.ceil(dt / cfl))
        nT = max(1, nT)

        # Scale A
        dtx = dt / nT / pv                       # timestep / pore volume
        B   = self._spdiags(dtx, 0) @ A          # A * dt/|Omega i|

        for _ in range(nT):
            Mw, Mo = self.RelPerm(S)             # compute mobilities
            fw = Mw / (Mw + Mo)                  # compute fractional flow
            S = S + (B@fw + fi*dtx)              # update saturation
        return S

    # NewtRaph() -- listing 10
    def saturation_step_implicit(self, S, V, dt, nNewtonMax=10, nTmax_log2=10):
        """Implicit FV discretisation of conserv. of mass (water sat.)."""
        A  = self.upwind_diff(V)                 # FV discretized transport operator
        pv = self.h2 * self.por.ravel()          # Pore volume = cell.vol * por
        fi = self._Q.clip(min=0)                 # Well inflow

        # For each iter, halve the sub/local dt
        for nT_log2 in range(0, nTmax_log2):
            nT = 2**nT_log2

            # Scale A
            dtx = dt / nT / pv                   # timestep / pore volume
            B   = self._spdiags(dtx, 0) @ A      # A * dt/|Omega i|

            Sn = S
            for _ in range(nT):
                Sp = Sn
                for _ in range(nNewtonMax):
                    Mw, Mo   = self.RelPerm(Sn)    # mobilities
                    dMw, dMo = self.dRelPerm(Sn)   # their derivatives
                    df = dMw/(Mw+Mo) - Mw/(Mw+Mo)**2 * (dMw + dMo)        # df w/ds
                    dG = sparse.eye(self.Nxy) - B @ self._spdiags(df, 0)  # deriv of G

                    fw = Mw / (Mw+Mo)               # fract. flow
                    G  = Sn - Sp - (B@fw + fi*dtx)  # G(s)
                    dS = spsolve(dG, G)             # compute dS
                    Sn = Sn - dS                    # update S

                    if np.sqrt(sum(dS**2)) < 1e-3:
                        # If converged: halt Newton iterations
                        break
                else:
                    # If never converged: increase nT, restart time loop
                    break
            else:
                # If completed all time steps, halt
                break
        else:
            # Failed (even with max nT) to complete all time steps
            print("Warning: did not converge")

        return Sn

    def time_stepper(self, dt, implicit=False):
        """Get ODE solver (integrator) for model.

        Whatever time step `dt` is given, both schemes will use smaller steps internally.

        - `explicit`: computes sub-`dt` based on CFL esitmate.
        - `implicit`: reduces sub-`dt` until convergence is achieved.
        """
        def integrate(S, k):
            self._set_Q(k)
            [P, V] = self.pressure_step(S)
            if implicit:
                S = self.saturation_step_implicit(S, V, dt)
            else:
                S = self.saturation_step_upwind(S, V, dt)
            return S
        return integrate


def recurse(fun, nSteps, x0, pbar=True, leave=True):
    """Recursively apply `fun` `nSteps` times.

    .. note:: `output[0] == x0`, hence `len(output) = nSteps + 1`.

    .. note::
        "Recurse" is a fancy programming term referring to a function calling itself.
        Here we implement it simply by a for loop, passing previous output as next intput.
        Indeed "recursive" is also an accurate description of causal (Markov) processes,
        such as nature or its simulators, which build on themselves.
    """
    # Init
    xx = np.zeros((nSteps+1,)+x0.shape)
    xx[0] = x0

    # Recurse
    kk = np.arange(nSteps)
    if pbar:
        kk = tqdm(kk, "Simulation", leave=leave, mininterval=1e-2)
    for k in kk:
        xx[k+1] = fun(xx[k], k)
    return xx
