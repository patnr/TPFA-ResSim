""".. include:: README.md"""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from struct_tools import DotDict, NicePrint
from tqdm.auto import tqdm

from TPFA_ResSim.grid import Grid2D
from TPFA_ResSim.plotting import Plot2D


@dataclass
class ResSim(NicePrint, Grid2D, Plot2D):
    """Reservoir simulator class.

    Implemented with OOP (instead of passing around dicts) to facilitate
    bookkeeping of ensemble forecasting
    (where parameter values of one instance should not influence another)

    Example:
    >>> model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64)
    >>> model.inj_xy=[[0, .32]]
    >>> model.prd_xy=[[1, 1]]
    >>> model.inj_rates=[[1]]
    >>> model.prd_rates=[[1]]
    >>> water_sat0 = np.zeros(model.Nxy)
    >>> dt = .35
    >>> nSteps = 2
    >>> S = model.sim(dt, nSteps, water_sat0, pbar=False)

    This produces the following values (used for automatic testing):
    >>> S[-1, [100, 1300, 2900]]
    array([0.9429345 , 0.91358172, 0.71554613])
    """
    # Dont use dataclass repr
    __repr__ = NicePrint.__repr__
    __str__ = NicePrint.__str__

    def __post_init__(self):
        defaults = dict(K=np.ones((2, *self.shape)),
                        por=np.ones(self.shape))
        for k, v in defaults.items():
            if getattr(self, k) is None:
                setattr(self, k, v)

    # Prefer __setattr__ approach (over @property get/set-ers)
    # because @property requires the _private pattern,
    # which is pretty ugly with dataclasses,
    # and also because can unify treatment of inj/prd wells.
    def __setattr__(self, key, val):
        if val is not None:
            # Well positions -- collocate at some node
            if key in ["inj_xy", "prd_xy"]:
                val = np.array(val, float).reshape((-1, 2))
                for i, (x, y) in enumerate(val):
                    val[i] = self.ind2xy(self.xy2ind(x, y))
            # Well rates
            if key in ["inj_rates", "prd_rates"]:
                nWell = len(getattr(self, key.replace("rates", "xy")))
                val = np.array(val, float).reshape((nWell, -1))
            # Permeabilities
            if key == "K":
                if np.isscalar(val):
                    val = np.full_like(self.shape, val, dtype=float)
                if val.size == self.size:
                    val = np.stack([val, val])  # both components
                val = val.reshape((2, *self.shape))
        # Set
        super().__setattr__(key, val)

    name: str = "Unnamed"
    """Description."""

    vw: float = 1.
    """Viscosity for water."""
    vo: float = 1.
    """Viscosity for oil."""
    swc: float = 0.
    """Irreducible saturation, water."""
    sor: float = 0.
    """Irreducible saturation, oil."""
    K: np.ndarray = None
    """Permeabilities (in x and y directions). Array of shape `(2, Nx, Ny)`)."""
    por: np.ndarray = None
    """Porosity; Array of shape `(Nx, Ny)`)."""

    nInj  = property(lambda self: len(self.inj_xy))
    """Num. of injector wells."""
    nPrd = property(lambda self: len(self.prd_xy))
    """Num. of producer wells."""

    inj_xy: np.ndarray = None
    """Array of shape `(nWell, 2)` of x- and y-coords for `nWell` injectors.

    Values should be betwen `0` and `Lx` or `Ly`.

    .. warning:: The wells get co-located with grid nodes, ref `xy2sub`.
        This is a design choice, not a mathematical necessity.
        An alternative would be to distribute them over nearby nodes.
    """
    prd_xy: np.ndarray = None
    """Like `inj_xy`, but for producing wells."""
    inj_rates: np.ndarray = None
    """Array of shape `(nWell, nTime)` -- or `(nWell, 1)` if constant-in-time.

    .. note:: Both `inj_rates` and `prd_rates` are rates should be positive.
        At each time index, it is asserted that the difference of their sums is 0,
        otherwise the model would silently input deficit from SW corner.
    """
    prd_rates: np.ndarray = None
    """Like `prd_rates`, but for producing wells."""

    def _set_Q(self, S, k):
        """Populate (for time `k`) the source/sink *field*, `Q`, from well specs."""
        Q = np.zeros(self.Nxy)
        rates = self.dynamic_rate(S, k)
        for kind in ['inj', 'prd']:
            # Populate Q
            xys = getattr(self, f'{kind}_xy')
            sgn = +1 if kind == "inj" else -1
            for xy, q in zip(xys, rates[kind]):
                Q[self.xy2ind(*xy)] += sgn * q  # += enables superimposition
            # Store the computed/dynamic rates
            if hasattr(self, "actual_rates"):
                self.actual_rates[kind][:, k] = rates[kind]
        self._Q = Q

    def _wanted_rates_at(self, k):
        """Lookup nominal/specified rates. Allows constant-in-time (singleton) spec."""
        get_now = lambda arr: np.copy(arr[k] if (len(arr) > 1) else arr[0])
        return (get_now(self.inj_rates.T),
                get_now(self.prd_rates.T))

    def dynamic_rate(self, S, k):
        """Compute the `actual_rates` for time index `k`.

        This default implementation simply reads the given well specifications.
        But you can overwrite (patch/inherit) it, for example to halt production wells
        if water saturation is too high or simply if the suggested rate is near 0.
        """
        inj, prd = self._wanted_rates_at(k)
        return dict(inj=inj, prd=prd)

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
    def estimate_1CFL(self, pv, V, fi):
        """Estimate 1/CFL for use with `saturation_step_upwind`."""
        # In-/Out-flux x-/y- faces
        XP = V.x.clip(min=0)
        XN = V.x.clip(max=0)
        YP = V.y.clip(min=0)
        YN = V.y.clip(max=0)
        Vi = XP[:-1, :] + YP[:, :-1] - XN[1:, :] - YN[:, 1:]

        flx = max((Vi.ravel() + fi) / pv)  # estimate of influx
        sat = self.swc + self.sor
        return 3 / (1 - sat) * flx  # NB: 3-->2 since no z-dim ?

    # Upstream() -- listing 8
    def saturation_step_upwind(self, S, V, dt):
        """Explicit upwind FV discretisation of conserv. of mass (water sat.)."""
        A  = self.upwind_diff(V)                 # FV discretized transport operator
        pv = self.h2 * self.por.ravel()          # Pore volume = cell volume * porosity
        fi = self._Q.clip(min=0)                 # Well inflow

        # Compute sub/local dt
        cfl1 = self.estimate_1CFL(pv, V, fi)
        nT = int(np.ceil(dt * cfl1))
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
            self._set_Q(S, k)

            # Catch some common issues before they become mysterious/insidious
            # (e.g. mass imblance silently inserts deficit in SW corner).
            assert len(self.inj_rates) == len(self.inj_xy)
            assert len(self.prd_rates) == len(self.prd_xy)
            assert np.isclose(self._Q.sum(), 0), "(inj - prd) does not sum to 0"
            assert np.all(self.inj_rates >= 0)
            assert np.all(self.prd_rates >= 0)
            assert np.all((0 <= self.K  ) & np.isfinite(self.K))
            assert np.all((0 <= self.por) & (self.por <= 1))

            [P, V] = self.pressure_step(S)
            if implicit:
                S = self.saturation_step_implicit(S, V, dt)
            else:
                S = self.saturation_step_upwind(S, V, dt)
            return S
        return integrate

    def sim(self, dt, nSteps, x0, pbar=True, leave=True, **kwargs):
        """Recursively (`nSteps` times) apply `time_stepper` with `dt`, from `x0`.

        .. note:: `output[0] == x0`, hence `len(output) = nSteps + 1`.

        .. note::
            "Recurse" is a describes a function calling itself.
            Here we implement it simply by a for loop.
            "Wecursive" is also an accurate description of causal (Markov) processes,
            such as nature or its simulators, which build on themselves.
        """
        step = self.time_stepper(dt, **kwargs)

        # pbar
        kk = np.arange(nSteps)
        if pbar:
            kk = tqdm(kk, "Simulation", leave=leave, mininterval=1e-2)

        # Init
        xx = np.zeros((nSteps+1,)+x0.shape)
        xx[0] = x0
        self.actual_rates = dict(inj=np.zeros((self.nInj, nSteps)),
                                 prd=np.zeros((self.nPrd, nSteps)))

        # Recurse
        for k in kk:
            xx[k+1] = step(xx[k], k)

        return xx
