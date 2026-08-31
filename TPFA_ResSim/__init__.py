""".. include:: README.md"""

from dataclasses import dataclass
from typing import Any, Callable

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
    >>> S, P = model.sim(dt, nSteps, water_sat0, pbar=False)

    This produces the following values (used for automatic testing):
    >>> S[-1, [100, 1300, 2900]]
    array([0.9429345 , 0.91358172, 0.71554613])
    """

    # Dont use dataclass repr
    __repr__ = NicePrint.__repr__
    __str__ = NicePrint.__str__

    def __post_init__(self) -> None:
        defaults = dict(K=np.ones((2, *self.shape)), por=np.ones(self.shape))
        for k, v in defaults.items():
            if getattr(self, k) is None:
                setattr(self, k, v)

    # Prefer __setattr__ approach (over @property get/set-ers)
    # because @property requires the _private pattern,
    # which is pretty ugly with dataclasses,
    # and also because can unify treatment of inj/prd wells.
    def __setattr__(self, key: str, val: Any) -> None:
        if val is not None:
            # Well positions -- collocate at some node
            if key in ["inj_xy", "prd_xy"]:
                val = np.array(val, float).reshape((-1, 2))
                for i, (x, y) in enumerate(val):
                    val[i] = self.ind2xy(self.xy2ind(x, y))
            # Well rates and/or pressures
            if key in ["inj_rates", "prd_rates", "inj_bhp", "prd_bhp"]:
                kind, spec = key.split("_")
                nWell = len(getattr(self, f"{kind}_xy"))
                val = np.array(val, float).reshape((nWell, -1))
            # Well indices
            if key in ["inj_WI", "prd_WI"]:
                nWell = len(getattr(self, key.replace("WI", "xy")))
                val = np.broadcast_to(np.ravel(np.asarray(val, float)), nWell).copy()
            # Permeabilities
            if key == "K":
                if np.isscalar(val):
                    val = np.full(self.shape, val, dtype=float)
                if val.size == self.size:
                    val = np.stack([val, val])  # both components
                val = val.reshape((2, *self.shape))
        # Set
        super().__setattr__(key, val)

    name: str = "Unnamed"
    """Description."""

    vw: float = 1.0
    """Viscosity for water."""
    vo: float = 1.0
    """Viscosity for oil."""
    swc: float = 0.0
    """Irreducible saturation, water."""
    sor: float = 0.0
    """Irreducible saturation, oil."""
    ct: float = 0.0
    """Total (rock + fluids) compressibility. The default, `0`, yields the
    incompressible model, whose pressure eqn. is elliptic
    (infinite speed of propagation), requires balanced source/sink terms,
    and only defines pressure up to an additive constant.

    Setting `ct > 0` ("slightly compressible" model) makes the pressure eqn.
    parabolic, $ φ \\, c_t \\, ∂p/∂t - ∇ ⋅ (K λ(s) ∇p) = q $,
    discretized by backward Euler over the same `dt` as the saturation step.
    Then total injection and production rates need not balance
    (storage absorbs the imbalance), enabling e.g. primary depletion.

    .. note:: The corresponding term of the *transport* (saturation) equation
        is included as well. Since the total velocity is no longer
        divergence-free, $ ∇ ⋅ v = q - φ \\, c_t \\, ∂p/∂t $, the water eqn. reads
        $$ φ \\, ∂s/∂t + s \\, φ \\, c_t \\, ∂p/∂t + ∇ ⋅ (f_w \\, v) = q_w $$
        i.e. the storage is charged to the phases in proportion to their
        saturation -- ref `storage_rate`. This is what makes the water and oil
        equations sum to the pressure equation, so that e.g. depleting a fully
        water-saturated reservoir leaves $s = 1$, rather than conjuring oil out
        of the produced volume.
        NB: deriving each phase equation individually would instead charge the
        water $ s \\, (c_r + c_w) \\, φ \\, ∂p/∂t $; the proportional split
        coincides with that iff $ c_w = c_o $, and is the natural closure when
        only the lump-sum constant `ct` is known. The difference,
        $ O(s (1-s) (c_w - c_o)) $, is within the fidelity discussed below.

    .. warning:: The model remains a *slightly* compressible one, accurate only
        to $O(c_t)$: `ct` is a single constant, rather than the
        saturation-weighted sum $ c_r + s \\, c_w + (1-s) \\, c_o $ of the rock and
        phase compressibilities, and the densities in the fluxes and the well
        rates are treated as constant (so reservoir and surface volumes are not
        distinguished). Fidelity therefore requires $ c_t \\, Δp \\ll 1 $ -- and
        note that this is *not* a matter of choosing `ct` small: summing the
        rows of the pressure system (ref `tests/test_compressible.py`) gives
        $ c_t \\, Δ\\bar{p} = V_\\mathrm{voidage} / V_\\mathrm{pore} $, i.e. the
        expansion asked of the fluids is set by the *voidage* (production minus
        injection), whatever `ct` may be.
    """
    # NB: the array attributes are typed `Any` since `__setattr__` normalizes
    # whatever array-like (nested lists, scalars) is assigned to them.
    K: Any = None
    """Permeabilities (in x and y directions). Array of shape `(2, Nx, Ny)`)."""
    por: Any = None
    """Porosity; Array of shape `(Nx, Ny)`)."""

    nInj = property(lambda self: len(self.inj_xy))
    """Num. of injector wells."""
    nPrd = property(lambda self: len(self.prd_xy))
    """Num. of producer wells."""

    inj_xy: Any = None
    """Array of shape `(nWell, 2)` of x- and y-coords for `nWell` injectors.

    Values should be betwen `0` and `Lx` or `Ly`.

    .. warning:: The wells get co-located with grid nodes, ref `xy2sub`.
        This is a design choice, not a mathematical necessity.
        An alternative would be to distribute them over nearby nodes.
    """
    prd_xy: Any = None
    """Like `inj_xy`, but for producing wells."""
    inj_rates: Any = None
    """Array of shape `(nWell, nTime)` -- or `(nWell, 1)` if constant-in-time.

    .. note:: Both `inj_rates` and `prd_rates` are rates should be positive.
        At each time index, it is asserted that the difference of their sums is 0,
        otherwise the model would silently input deficit from SW corner.
    """
    prd_rates: Any = None
    """Like `prd_rates`, but for producing wells."""
    inj_WI: Any = None
    """Well indices (`None`, or an array of shape `(nWell,)`) for the injectors.

    Compute $ WI $ with `peaceman_WI`, or set it directly
    (it need not come from any particular formula).
    Without it, `actual_bhp` is `nan` and BHP control (`inj_bhp`) unavailable.

    The well index, $ WI $, is the *sub-grid* well model: the constant of
    proportionality relating a well's flow rate to $ \\Delta p $
    $$ q = WI \\, λ_t \\, \\Delta p \\,,$$
    with $ λ_t $ the total mobility (ref `RelPerm`) of the well's cell.
    Here the pressure difference is $ \\Delta p = p_\\mathrm{cell} - p_\\mathrm{bh} $
    known as the "drawdown", with sign reversed for an injector. Re-arranging,
    $$ p_\\mathrm{bh} = p_\\mathrm{cell} ∓ q / (WI \\, λ_t) \\,. $$

    NB: the drawdown cannot be replaced by plain cell pressure,
    because that is a cell average, which is highly sensitive to the
    chosen discretisation size around a point source/sink (an idealized,
    theoretical singularity).

    .. warning:: The drawdown is *not* a fixed offset that one could calibrate away once and for all.
        Being $ q / (WI \\, λ_t) $, it tracks the mobility -- which, in a waterflood,
        dips as the front arrives (by half, for equal viscosities). So the gap
        doubles at breakthrough: precisely when the well is most interesting.
    """
    prd_WI: Any = None
    """Like `inj_WI`, but for producing wells."""
    inj_bhp: Any = None
    """Bottom-hole pressures for the injectors. `None`, or an array shaped like
    `inj_rates`, i.e. `(nWell, nTime)` -- or `(nWell, 1)` if constant-in-time.

    A well whose entry is finite is **BHP-controlled** at that time. This is
    solved *simultaneously* with the pressure field (not lagged by a time step):
    `TPFA` puts $ WI λ_t $ on its diagonal and $ WI λ_t \\, p_\\mathrm{bh} $
    on its right-hand side.
    Only once $ p $ is known is the resulting rate folded into the source field,
    `_Q`, for the transport step -- ref `_set_Q` and `_realize_bhp`.

    Entries left as `nan` -- which is the default, for every well -- keep the
    well rate-controlled, per `inj_rates`. So the two mechanisms can be mixed
    freely, both across wells and in time.

    Requires `inj_WI`. The realized rates are recorded in `actual_rates`, and
    `inj_rates` is ignored (it may be left as `None`).

    .. note:: BHP control also *anchors the pressure*. With `ct == 0` the
        pressure equation is otherwise a pure-Neumann problem: solvable only up
        to a constant (which `TPFA` pins arbitrarily, ref article p. 13), and
        only if injection balances production. A single BHP well lifts both
        restrictions -- the level is then set by $ p_\\mathrm{bh} $, and the
        voidage by the well model.

    .. warning:: There is no switching of control modes. A producer whose
        $ p_\\mathrm{bh} $ rose above its cell pressure would *inject* (at the
        cell's fractional flow); rather than let that pass silently,
        `_realize_bhp` asserts against it. Rate control with a BHP limit -- the
        industrial default -- would mean iterating each well's mode within each
        step; `dynamic_rate` is the intended place to approximate it.
    """
    prd_bhp: Any = None
    """Like `inj_bhp`, but for producing wells."""

    def _set_Q(self, S: np.ndarray | None, k: int) -> None:
        """Populate (for time `k`) the source/sink *field*, `Q`, from well specs.

        Rate-controlled wells contribute their rate to `Q` directly.
        BHP-controlled ones (ref `inj_bhp`) cannot: their rate is not yet known,
        but instead contribute to the TPFA pressure equation system.
        The resulting rates are then folded into `Q` by `_realize_bhp`.
        """
        Q, bhp_diag, bhp_rhs = np.zeros((3, self.Nxy))
        rates = self.dynamic_rate(S, k)
        bhps = self._wanted_bhp_at(k)
        self._WI_lam = {}
        for kind in ["inj", "prd"]:
            sgn = +1 if kind == "inj" else -1
            inds = self._well_inds(kind)
            on_bhp = np.isfinite(bhps[kind])

            # The well model's constant of proportionality, WI * λ_t.
            # NB: `nan` marks the rate-controlled wells, throughout.
            WI_lam = np.full(len(inds), np.nan)
            if on_bhp.any():
                WI = getattr(self, f"{kind}_WI")
                assert WI is not None, f"BHP control requires `{kind}_WI`."
                assert S is not None, "BHP control requires `S` (for λ_t)."
                Mw, Mo = self.RelPerm(S)
                WI_lam[on_bhp] = WI[on_bhp] * (Mw + Mo)[inds[on_bhp]]
            self._WI_lam[kind] = WI_lam

            # Populate Q (or, for the BHP wells, the well model). += superimposes.
            for i, ind in enumerate(inds):
                if on_bhp[i]:
                    bhp_diag[ind] += WI_lam[i]
                    bhp_rhs[ind] += WI_lam[i] * bhps[kind][i]
                else:
                    Q[ind] += sgn * rates[kind][i]

            # Store the computed/dynamic rates.
            # Those of the BHP wells are only known to `_realize_bhp`.
            if hasattr(self, "actual_rates"):
                self.actual_rates[kind][:, k] = np.where(on_bhp, np.nan, rates[kind])
        self._Q, self._bhp_diag, self._bhp_rhs = Q, bhp_diag, bhp_rhs
        self._bhp_wanted = bhps

    def _realize_bhp(self, P: np.ndarray, k: int) -> None:
        """Fold the (now solved-for) rates of the BHP wells into `_Q`.

        Must run between `pressure_step` and the saturation step: the transport
        scheme reads `_Q` (in `upwind_diff`, `storage_rate`, `estimate_1CFL`),
        and until now it held only the rate-controlled wells.

        By construction of the linear system of `TPFA`, this leaves `_Q` equal
        to the *total* well flux, which is what keeps `storage_rate` -- and
        hence the transport step -- consistent with the pressure solution.
        """
        self._Q = self._Q + self._bhp_rhs - self._bhp_diag * P
        for kind in ["inj", "prd"]:
            WI_lam = self._WI_lam[kind]
            on_bhp = np.isfinite(WI_lam)
            if not on_bhp.any():
                continue
            inds = self._well_inds(kind)[on_bhp]
            sgn = +1 if kind == "inj" else -1
            q = sgn * WI_lam[on_bhp] * (self._bhp_wanted[kind][on_bhp] - P[inds])
            assert np.all(q > -1e-8 * (1 + np.abs(q).max())), (
                f"A BHP-controlled '{kind}' well would flow backwards, its"
                " `p_bh` having ended up on the wrong side of its cell pressure."
                " This model does not switch control modes; ref `inj_bhp`."
            )
            if hasattr(self, "actual_rates"):
                self.actual_rates[kind][on_bhp, k] = q

    def _at_time(self, arr: Any, k: int, nWell: int, absent: float) -> np.ndarray:
        """Lookup a `(nWell, nTime)` well spec at time `k`.

        Allows a constant-in-time (singleton) spec, and an unset (`None`) one,
        which yields `absent` -- `0` for a rate, `nan` for a BHP.
        """
        if arr is None:
            return np.full(nWell, absent)
        arr = arr.T
        return np.copy(arr[k] if (len(arr) > 1) else arr[0])

    def _wanted_rates_at(self, k: int) -> tuple:
        """Lookup nominal/specified rates. Allows constant-in-time (singleton) spec."""
        # fmt: off
        return (self._at_time(self.inj_rates, k, self.nInj, 0.),
                self._at_time(self.prd_rates, k, self.nPrd, 0.))
        # fmt: on

    def _wanted_bhp_at(self, k: int) -> dict:
        """Like `_wanted_rates_at`, but for `inj_bhp` (`nan` ⇒ rate-controlled)."""
        # fmt: off
        return dict(inj=self._at_time(self.inj_bhp, k, self.nInj, np.nan),
                    prd=self._at_time(self.prd_bhp, k, self.nPrd, np.nan))
        # fmt: on

    def _well_inds(self, kind: str) -> np.ndarray:
        """Flat indices of the cells holding the `kind` (`"inj"`/`"prd"`) wells."""
        xy = np.asarray(getattr(self, f"{kind}_xy"))
        return np.atleast_1d(self.xy2ind(*xy.T))

    def dynamic_rate(self, S: np.ndarray | None, k: int) -> dict:
        """Compute the `actual_rates` for time index `k`.

        This default implementation simply reads the given well specifications.
        But you can overwrite (patch/inherit) it, for example to halt production wells
        if water saturation is too high or simply if the suggested rate is near 0.
        """
        inj, prd = self._wanted_rates_at(k)
        return dict(inj=inj, prd=prd)

    def peaceman_WI(self, xy: Any, rw: float, skin: float = 0.0) -> np.ndarray:
        """Peaceman's well index for wells at `xy`, of radius `rw`.

        $$ WI = \\frac{2 π \\sqrt{k_x k_y}}{\\ln(r_e / r_w) + \\mathrm{skin}} $$

        where the *equivalent radius*, $ r_e $, is the distance from the well at
        which the (analytic, radial) pressure equals the (numerical) pressure of
        the well's cell:
        $$ r_e = 0.28 \\,
           \\frac{\\sqrt{\\sqrt{k_y/k_x} \\, h_x^2
                          + \\sqrt{k_x/k_y} \\, h_y^2}}
                  {(k_y/k_x)^{1/4} + (k_x/k_y)^{1/4}} \\,, $$
        which reduces to the familiar $ r_e = 0.198 \\, h $ on an isotropic,
        square grid. That constant is not a fudge factor: it is a property of
        the 5-point stencil that `TPFA` assembles, and this model reproduces it
        (`tests/test_wells.py` recovers $ r_e / h → 0.198 $ from the simulated
        drawdown, and thereby the analytic, radial well pressure to within 0.2%,
        on grids from 16² to 64²).

        >>> model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32)
        >>> model.peaceman_WI([[.5, .5]], rw=1e-3).round(4)
        array([3.4476])

        .. note:: There is no thickness factor because the model is 2D *areal*,
            i.e. of unit thickness -- as is already implicit in `TPFA`, whose
            transmissibilities read $ k \\, h_y / h_x $, and in `h2` serving as
            the cell volume.

        .. warning:: `rw` is a *physical length*, whereas the rest of the model
            is scale-free (ref the `Lx`, `K` and `vw` defaults). Only the ratio
            $ r_e / r_w $ enters, so this is consistent -- but it does mean that
            `rw` must be given in the same (arbitrary) length unit as `Lx`.
        """
        xy = np.asarray(xy, float).reshape((-1, 2))
        ix, iy = self.xy2sub(*xy.T)
        kx, ky = self.K[0][ix, iy], self.K[1][ix, iy]
        # fmt: off
        a, b = np.sqrt(ky/kx), np.sqrt(kx/ky)
        r_e  = .28 * np.sqrt(a*self.hx**2 + b*self.hy**2) / (a**.5 + b**.5)
        return 2*np.pi*np.sqrt(kx*ky) / (np.log(r_e/rw) + skin)
        # fmt: on

    def _crossings(self, p0: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Parameters $t ∈ [0, 1]$ at which `p0 + t*d` crosses a cell boundary."""
        ts = [0.0, 1.0]
        for ax, h in enumerate([self.hx, self.hy]):
            if d[ax] == 0:
                continue
            lo, hi = sorted([p0[ax], p0[ax] + d[ax]])
            ts += [
                (i * h - p0[ax]) / d[ax]
                for i in range(int(np.floor(lo / h)) + 1, int(np.ceil(hi / h)))
            ]
        return np.unique(np.clip(ts, 0, 1))  # NB: `unique` also sorts

    def well_path(self, vertices: Any, rw: float, skin: float = 0.0) -> tuple:
        """Discretize a well *path* (a polyline) into 1 weighted completion per traversed cell.

        Returns `(xy, WI, w)`:

        - `xy`: centres of the cells that the path traverses -- i.e. a value for
          `inj_xy`/`prd_xy`. Several completions act as a single well simply by
          being several wells: `_set_Q` superimposes them (with `+=`).
        - `WI`: their well indices, i.e. a value for `inj_WI`/`prd_WI`. Each is
          `peaceman_WI` for its cell, scaled by the fraction of that cell which
          the path actually traverses (so a cell merely clipped by the path
          contributes proportionally less).
        - `w`: `WI / WI.sum()`, for apportioning the rate among its completions:
          `inj_rates = rate * w[:, None]`.
          This is the standard (static) allocation -- proportional to the well index,
          hence to both the contacted length and the local permeability.

        >>> model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10)
        >>> xy, WI, w = model.well_path([[.05, .05], [.45, .05]], rw=1e-2)
        >>> xy.T[0]  # the traversed cells, in x
        array([0.05, 0.15, 0.25, 0.35, 0.45])
        >>> w  # the end cells, entered mid-way, get half the rate
        array([0.125, 0.25 , 0.25 , 0.25 , 0.125])

        .. note:: Under BHP control this `w` is exact (assuming 0 gravity and friction):
            the completions simply share a `p_bh`.
            Under *rate* control, `w` is an approximation unless the cell pressures are equal.
            Solving for it would make $ p_\\mathrm{bh} $ an extra
            unknown, i.e. a bordered linear system -- which the 5-diagonal
            assembly of `TPFA` (Listing 1) is not set up for. Use
            `dynamic_rate` to reallocate per step, if it matters.

        .. warning:: The completions are treated as independent *vertical* wells,
            which seems reasonable in a 2D areal model. Thus `nInj`/`nPrd` count completions, not wells.
        """
        V = np.asarray(vertices, float).reshape((-1, 2))
        assert len(V) >= 2, "A well path needs at least 2 vertices."
        # Walk the polyline, accumulating traversed length per cell
        lengths: dict = {}
        for p0, p1 in zip(V[:-1], V[1:]):
            d = p1 - p0
            L = float(np.hypot(*d))
            if L == 0:
                continue
            ts = self._crossings(p0, d)
            mids = p0 + np.outer((ts[:-1] + ts[1:]) / 2, d)
            for mid, dt in zip(mids, np.diff(ts)):
                sub = tuple(int(i) for i in self.xy2sub(*mid))
                lengths[sub] = lengths.get(sub, 0.0) + L * dt
        # Discard the slivers left by corner crossings
        total = sum(lengths.values())
        lengths = {k: v for k, v in lengths.items() if v > 1e-9 * total}

        subs = np.array(list(lengths))
        xy = self.sub2xy(*subs.T).T
        # Scale each WI by how much of its cell the path traverses, relative
        # to the cell size -- so an axis-aligned full crossing scores exactly 1
        # (and a diagonal one √2, it contacting that much more rock).
        frac = np.array(list(lengths.values())) / np.sqrt(self.h2)
        WI = frac * self.peaceman_WI(xy, rw, skin)
        return xy, WI, WI / WI.sum()

    def bhp(self, S: np.ndarray, P: np.ndarray, rates: dict) -> dict:
        """Bottom-hole pressures implied by `rates`, via the well indices.

        I.e. the well model of `inj_WI`, solved for $ p_\\mathrm{bh} $.
        Wells whose well index is `None` yield `nan`.

        `S` and `P` (both flat) should be the saturation and the pressure of the
        *same* `pressure_step`, i.e. `SS[k]` and `PP[k+1]` of `sim` -- which is
        what `actual_bhp` records, so prefer reading that.

        .. warning:: Unlike the cell pressure, this is (to the accuracy of the
            well model) independent of the grid resolution -- which is the whole
            point. But it inherits the model's premises: in particular $ λ_t $
            is that of the well's *cell*, so an injector's injectivity is
            governed by the mobility of whatever the cell currently holds,
            rather than by that of the injectant.
        """
        Mw, Mo = self.RelPerm(S)
        Mt = Mw + Mo
        out = {}
        for kind in ["inj", "prd"]:
            WI = getattr(self, f"{kind}_WI")
            if WI is None:
                out[kind] = np.full(len(getattr(self, f"{kind}_xy")), np.nan)
                continue
            ii = self._well_inds(kind)
            sgn = +1 if kind == "inj" else -1
            out[kind] = P[ii] + sgn * rates[kind] / (WI * Mt[ii])
        return out

    # Pres() -- listing 5
    def pressure_step(
        self,
        S: np.ndarray,
        P: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple:
        """Compute permeabilities then solve Darcy's equation. Returns `[P, V]`.

        `P` (flat, like `S`) is the *previous* step's pressure: used (and
        required) only if `ct > 0`, along with `dt`. The new one replaces it.
        """
        # Compute K*λ(S)
        Mw, Mo = self.RelPerm(S)
        Mt = Mw + Mo
        Mt = Mt.reshape(self.shape)
        KM = Mt * self.K
        # Compute pressure and extract fluxes
        [P, V] = self.TPFA(KM, P, dt)  # NB: updates `P`
        return P, V

    def _spdiags(self, data: Any, diags: Any) -> sparse.dia_matrix:
        return sparse.spdiags(data, diags, self.Nxy, self.Nxy)

    def rescale_sat(self, s: np.ndarray) -> np.ndarray:
        """Account for irreducible saturations. Ref paper, p. 32."""
        return (s - self.swc) / (1 - self.swc - self.sor)

    # RelPerm() -- listing 6
    def RelPerm(self, s: np.ndarray) -> tuple:
        """Rel. permeabilities of oil and water. Return as mobilities (perm/viscocity)."""
        S = self.rescale_sat(s)
        Mw = S**2 / self.vw  # Water mobility
        Mo = (1 - S) ** 2 / self.vo  # Oil mobility
        return Mw, Mo

    def dRelPerm(self, s: np.ndarray) -> tuple:
        """Derivatives of `RelPerm`."""
        S = self.rescale_sat(s)
        dMw = 2 * S / self.vw / (1 - self.swc - self.sor)
        dMo = -2 * (1 - S) / self.vo / (1 - self.swc - self.sor)
        return dMw, dMo

    # TPFA() -- Listing 1
    def TPFA(
        self,
        K: np.ndarray,
        P: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple:
        """Two-point flux-approximation (TPFA) of Darcy: $ -∇(K ∇u) = q $

        i.e. steady-state diffusion w/ nonlinear coefficient, $K$,
        if `ct == 0`. Otherwise (slightly compressible model) solve
        the backward-Euler step of $ φ c_t ∂u/∂t - ∇(K ∇u) = q $,
        which requires the previous pressure, `P`, and `dt`.

        After solving for pressure `P`, extract the fluxes `V`
        by finite differences.
        """
        # Compute transmissibilities by harmonic averaging.
        L = 1 / K
        TX = np.zeros((self.Nx + 1, self.Ny))
        TY = np.zeros((self.Nx, self.Ny + 1))
        TX[1:-1, :] = 2 * self.hy / self.hx / (L[0, :-1, :] + L[0, 1:, :])
        TY[:, 1:-1] = 2 * self.hx / self.hy / (L[1, :, :-1] + L[1, :, 1:])

        # Assemble TPFA discretization matrix.
        x1 = TX[:-1, :].ravel()
        x2 = TX[1:, :].ravel()
        y1 = TY[:, :-1].ravel()
        y2 = TY[:, 1:].ravel()

        # Setup linear system
        DiagVecs = [-x2, -y2, y1 + y2 + x1 + x2, -y1, -x1]
        DiagIndx = [-self.Ny, -1, 0, 1, self.Ny]
        q = self._Q
        if self.ct > 0:
            # Accumulation term (φ ct h²/dt) of backward Euler.
            # Renders the system nonsingular (unlike the pure-Neumann problem).
            assert P is not None and dt is not None, (
                "Compressible model (ct > 0) requires the previous P, and dt."
            )
            accum = self.por.ravel() * self.ct * self.h2 / dt
            DiagVecs[2] = DiagVecs[2] + accum
            q = q + accum * P
        elif not self._bhp_diag.any():
            # Pin the (otherwise pure-Neumann, hence singular) problem.
            # Unnecessary -- and wrong -- if a BHP well already anchors it.
            DiagVecs[2][0] += np.sum(self.K[:, 0, 0])  # ref article p. 13
        # Well model of the BHP-controlled wells (`0` if there are none)
        DiagVecs[2] = DiagVecs[2] + self._bhp_diag
        q = q + self._bhp_rhs
        A = self._spdiags(DiagVecs, DiagIndx)

        # Solve; compute A\q to update P
        # P = np.linalg.solve(A.A, q) # direct dense solver
        P = spsolve(A.tocsr(), q)  # direct sparse solver.
        # P, _info = cg(A, q)         # conjugate gradient
        # Could also try scipy.linalg.solveh_banded which, according to
        # https://scicomp.stackexchange.com/a/30074 uses the Thomas algorithm,
        # as recommended by Aziz and Settari ("Petro. Res. simulation").
        # NB: stackexchange also mentions that solve_banded does not work well
        # when the band offsets large, i.e. higher-dimensional problems.

        # Extract fluxes, via a grid-shaped view of the (flat) pressure.
        P2d = P.reshape(self.shape)
        # `Any` coz ty cannot see that `DotDict` provides attribute access to keys
        V: Any = DotDict(
            x=np.zeros((self.Nx + 1, self.Ny)),
            y=np.zeros((self.Nx, self.Ny + 1)),
        )
        V.x[1:-1, :] = (P2d[:-1, :] - P2d[1:, :]) * TX[1:-1, :]
        V.y[:, 1:-1] = (P2d[:, :-1] - P2d[:, 1:]) * TY[:, 1:-1]
        return P, V

    # GenA() -- listing 7
    def upwind_diff(self, V: Any) -> sparse.dia_matrix:
        """Upwind finite-volume scheme."""
        fp = self._Q.clip(max=0)  # production
        # Flow fluxes, separated into direction (x-y) and sign
        x1 = V.x.clip(max=0)[:-1, :].ravel()
        y1 = V.y.clip(max=0)[:, :-1].ravel()
        x2 = V.x.clip(min=0)[1:, :].ravel()
        y2 = V.y.clip(min=0)[:, 1:].ravel()
        DiagVecs = [x2, y2, fp + y1 - y2 + x1 - x2, -y1, -x1]
        DiagIndx = [-self.Ny, -1, 0, 1, self.Ny]
        A = self._spdiags(DiagVecs, DiagIndx)
        return A

    def storage_rate(self, V: Any) -> np.ndarray:
        """The volume rate, per cell, that goes into storage: $ q - ∇ ⋅ V $.

        For the incompressible model this is `0`: the fluxes balance the wells
        exactly, cell by cell. With `ct > 0` it is (by construction of the
        linear system of `TPFA`) the accumulation term of the backward-Euler
        step, $ φ \\, c_t \\, h^2 \\, (p^{n+1} - p^n) / Δt $, which the saturation
        steps charge to the phases (ref `ct`).

        Computing it from `V` (rather than from $p^{n+1} - p^n$) means it is
        *exactly* the imbalance seen by the transport scheme, whose `upwind_diff`
        is assembled from the same fluxes.
        """
        if self.ct == 0:
            return np.zeros(self.Nxy)
        divV = (V.x[1:, :] - V.x[:-1, :]) + (V.y[:, 1:] - V.y[:, :-1])
        return self._Q - divV.ravel()

    # Extracted from Upstream()
    def estimate_1CFL(self, pv: np.ndarray, V: Any, fi: np.ndarray) -> float:
        """Estimate 1/CFL for use with `saturation_step_upwind`."""
        # In-/Out-flux x-/y- faces
        XP = V.x.clip(min=0)
        XN = V.x.clip(max=0)
        YP = V.y.clip(min=0)
        YN = V.y.clip(max=0)
        Vi = XP[:-1, :] + YP[:, :-1] - XN[1:, :] - YN[:, 1:]

        flx = max((Vi.ravel() + fi) / pv)  # estimate of influx
        # NB: `storage_rate` is not counted here. In practice it is a small
        # fraction of the fluxes that are (under 20% even at `ct = 10`),
        # so the safety factor below covers it.
        sat = self.swc + self.sor
        return 3 / (1 - sat) * flx  # NB: 3-->2 since no z-dim ?

    # Upstream() -- listing 8
    def saturation_step_upwind(self, S: np.ndarray, V: Any, dt: float) -> np.ndarray:
        """Explicit upwind FV discretisation of conserv. of mass (water sat.)."""
        # fmt: off
        A  = self.upwind_diff(V)                 # FV discretized transport operator
        pv = self.h2 * self.por.ravel()          # Pore volume = cell volume * porosity
        fi = self._Q.clip(min=0)                 # Well inflow
        st = self.storage_rate(V)                # Storage (0 if incompressible)

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
            S = S + (B@fw + (fi - S*st)*dtx)     # update saturation
        # fmt: on
        return S

    # NewtRaph() -- listing 10
    def saturation_step_implicit(
        self,
        S: np.ndarray,
        V: Any,
        dt: float,
        nNewtonMax: int = 10,
        nTmax_log2: int = 10,
    ) -> np.ndarray:
        """Implicit FV discretisation of conserv. of mass (water sat.).

        .. warning:: Far outside the $ c_t \\, Δp \\ll 1 $ regime (ref `ct`),
            the Newton iteration can converge -- silently -- to a spurious root
            of the residual, outside $[0, 1]$: the polynomial `RelPerm` extends
            smoothly beyond the unit interval, and the sub-`dt` halving only
            triggers on *non*-convergence. The explicit scheme
            (`saturation_step_upwind`), being monotone, stays within $[0, 1]$
            even for extreme `ct`.
        """
        # fmt: off
        A  = self.upwind_diff(V)                 # FV discretized transport operator
        pv = self.h2 * self.por.ravel()          # Pore volume = cell.vol * por
        fi = self._Q.clip(min=0)                 # Well inflow
        st = self.storage_rate(V)                # Storage (0 if incompressible)

        # For each iter, halve the sub/local dt
        for nT_log2 in range(0, nTmax_log2):
            nT = 2**nT_log2

            # Scale A
            dtx = dt / nT / pv                   # timestep / pore volume
            B   = self._spdiags(dtx, 0) @ A      # A * dt/|Omega i|
            C   = self._spdiags(dtx*st, 0)       # storage, likewise scaled

            Sn = S
            for _ in range(nT):
                Sp = Sn
                for _ in range(nNewtonMax):
                    Mw, Mo   = self.RelPerm(Sn)    # mobilities
                    dMw, dMo = self.dRelPerm(Sn)   # their derivatives
                    df = dMw/(Mw+Mo) - Mw/(Mw+Mo)**2 * (dMw + dMo)        # df w/ds
                    dG = (sparse.eye(self.Nxy) + C                        # deriv of G
                          - B @ self._spdiags(df, 0))

                    fw = Mw / (Mw+Mo)               # fract. flow
                    G  = Sn - Sp - (B@fw + (fi - Sn*st)*dtx)  # G(s)
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
        # fmt: on

        return Sn

    def time_stepper(self, dt: float, implicit: bool = False) -> Callable:
        """Get ODE solver (integrator) for model.

        Whatever time step `dt` is given, both schemes will use smaller steps internally.

        - `explicit`: computes sub-`dt` based on CFL esitmate.
        - `implicit`: reduces sub-`dt` until convergence is achieved.
        """

        def integrate(S, P, k):
            self._set_Q(S, k)

            # Catch some common issues before they become mysterious/insidious
            # (e.g. mass imblance silently inserts deficit in SW corner).
            for kind in ["inj", "prd"]:
                rates = getattr(self, f"{kind}_rates")
                if rates is not None:  # `None` ⇒ purely BHP-controlled
                    assert len(rates) == len(getattr(self, f"{kind}_xy"))
                    assert np.all(rates >= 0)
            if self.ct == 0 and not self._bhp_diag.any():
                # Incompressible ⇒ no storage ⇒ src/sinks must balance.
                # Unless a BHP well absorbs the imbalance, ref `inj_bhp`.
                assert np.isclose(self._Q.sum(), 0), "(inj - prd) does not sum to 0"
            assert np.all((0 <= self.K) & np.isfinite(self.K))
            assert np.all((0 <= self.por) & (self.por <= 1))

            [P, V] = self.pressure_step(S, P, dt)  # NB: `P` in ⇒ old, out ⇒ new
            self._realize_bhp(P, k)
            if hasattr(self, "actual_bhp"):
                now = {kd: self.actual_rates[kd][:, k] for kd in ["inj", "prd"]}
                for kd, p_bh in self.bhp(S, P, now).items():
                    self.actual_bhp[kd][:, k] = p_bh
            if implicit:
                S = self.saturation_step_implicit(S, V, dt)
            else:
                S = self.saturation_step_upwind(S, V, dt)
            return S, P

        return integrate

    def sim(
        self,
        dt: float,
        nSteps: int,
        S0: np.ndarray,
        P0: np.ndarray | None = None,
        pbar: bool = True,
        leave: bool = True,
        **kwargs,
    ) -> tuple:
        """Recursively (`nSteps` times) apply `time_stepper` with `dt`, from `S0`.

        Returns the saturation and pressure trajectories, `(SS, PP)`.

        .. note:: `SS[0] == S0` and `PP[0] == P0`, hence both have `len = nSteps + 1`.
            `P0` defaults to zeros. It is only consequential if `ct > 0`.
        """
        step = self.time_stepper(dt, **kwargs)

        # pbar
        kk = np.arange(nSteps)
        if pbar:
            kk = tqdm(kk, "Simulation", leave=leave, mininterval=1e-2)

        # Allocate
        SS = np.zeros((nSteps + 1,) + S0.shape)
        PP = np.zeros((nSteps + 1, self.Nxy))
        # fmt: off
        self.actual_rates = dict(inj=np.zeros((self.nInj, nSteps)),
                                 prd=np.zeros((self.nPrd, nSteps)))
        self.actual_bhp   = dict(inj=np.full((self.nInj, nSteps), np.nan),
                                 prd=np.full((self.nPrd, nSteps), np.nan))
        # fmt: on

        # Init
        SS[0] = S0
        if P0 is not None:
            PP[0] = P0

        # Recurse
        for k in kk:
            SS[k + 1], PP[k + 1] = step(SS[k], PP[k], k)

        return SS, PP
