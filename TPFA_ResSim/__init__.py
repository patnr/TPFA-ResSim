""".. include:: README.md"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm.auto import tqdm

from TPFA_ResSim._repr import AlignedRepr
from TPFA_ResSim.grid import Grid2D
from TPFA_ResSim.plotting import Plot2D
from TPFA_ResSim.wells import Wells, peaceman_WI, well_path  # noqa: F401


class Fluxes(NamedTuple):
    """The (discrete Darcy) fluxes through the cell faces, from `ResSim.TPFA`.

    Positive is in the direction of increasing index. The fluxes through the
    *boundary* faces are `0`: the reservoir is closed (no-flow) all around.
    """

    x: np.ndarray
    """Fluxes through the x-normal faces. Shape `(Nx+1, Ny)`."""
    y: np.ndarray
    """Fluxes through the y-normal faces. Shape `(Nx, Ny+1)`."""


@dataclass
class ResSim(AlignedRepr, Grid2D, Plot2D):
    """Reservoir simulator class.

    Implemented with OOP (instead of passing around dicts) to facilitate
    bookkeeping of ensemble forecasting
    (where parameter values of one instance should not influence another)

    Example:
    >>> model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64, wells=[
    ...     dict(xy=[0, .32], rate=+1),   # injector
    ...     dict(xy=[1, 1], rate=-1),     # producer
    ... ])
    >>> water_sat0 = np.zeros(model.Nxy)
    >>> dt = .35
    >>> nSteps = 2
    >>> S, P = model.sim(dt, nSteps, water_sat0, pbar=False)

    This produces the following values (used for automatic testing):
    >>> S[-1, [100, 1300, 2900]]
    array([0.9429345 , 0.91358172, 0.71554613])
    """

    # Dont use dataclass repr
    __repr__ = AlignedRepr.__repr__

    # Prefer __setattr__ approach (over @property get/set-ers)
    # because @property requires the _private pattern,
    # which is pretty ugly with dataclasses.
    def __setattr__(self, key: str, val: Any) -> None:
        # Defaults that the dataclass cannot express, depending as they do on the grid
        if val is None:
            if key == "K":
                val = np.ones((2, *self.shape))
            elif key == "por":
                val = np.ones(self.shape)
        # Permeabilities
        if key == "K" and val is not None:
            if np.isscalar(val):
                val = np.full(self.shape, val, dtype=float)
            if val.size == self.size:
                val = np.stack([val, val])  # both components
            val = val.reshape((2, *self.shape))
        # Wells -- records (or `None`) get assembled into a `Wells`, which then
        # gets bound, whereupon it snaps its completions onto this grid.
        # NB: the wells' own normalization is `TPFA_ResSim.wells.Wells.__setattr__`.
        if key == "wells":
            if not isinstance(val, Wells):
                val = Wells.from_records(self, val)
            val._bind(self)
        # Set
        super().__setattr__(key, val)

    name: str = "Unnamed"
    """Description."""

    cdarcy: float = 1.0
    """Unit conversion factor for Darcy's law, $C$ -- ECLIPSE's `CDARCY`.

    If you want to change unit system you not only need to manually convert
    the dimensional input quantities to the new units, but also change $C$ according to
    $$ C = \\frac{u_k \\, u_p \\, u_t}{u_μ \\, u_L^2} \\,, $$
    (with $u_k$ the SI magnitude of the unit chosen for $k$).
    Any *coherent* system gives `1`: choose base units for length, time and mass,
    derive $u_p = M/(L T^2)$, $u_μ = M/(L T)$ and $u_k = L^2$ from them.

    | System | $u_L$ | $u_t$ | $u_p$ | $u_k$ | $u_μ$ | rate | $C$ |
    |---|---|---|---|---|---|---|---|
    | SI | m | s | Pa | m² | Pa·s | m²/s | `1` |
    | CGS | cm | s | barye | cm² | poise | cm²/s | `1` |
    | MTS | m | s | pièze | m² | pz·s | m²/s | `1` |
    | mm-ms-g | mm | ms | MPa | mm² | kPa·s | mm²/ms | `1` |
    | Darcy's own | cm | s | atm | darcy | cP | cm²/s | `1` |
    | metric | m | day | bar | mD | cP | m²/day | `0.008527` |
    | field-like | ft | day | psi | mD | cP | ft²/day | `0.006328` |
    | lab | cm | hour | atm | mD | cP | cm²/hour | `3.6` |

    .. note:: The rate unit is forced to $u_L^2/u_t$ -- an areal rate.

        A well rate of `20` for a 25 m thick reservoir means 500 m³/day.

    .. note:: $C$ enters at exactly 2 sites, both of them Darcy's law.

        The transmissibilities of `TPFA` and the well index of
        `TPFA_ResSim.wells.peaceman_WI`. Everything else is derivative, and
        already consistent.
    """

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

    .. note:: The corresponding term of the *transport* equation is included too.

        Since the total velocity is no longer divergence-free,
        $ ∇ ⋅ v = q - φ \\, c_t \\, ∂p/∂t $, the water eqn. reads
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

    .. warning:: The model remains only a *slightly* compressible one.

        It is accurate to $O(c_t)$ alone: `ct` is a single constant, rather
        than the saturation-weighted sum $ c_r + s \\, c_w + (1-s) \\, c_o $ of
        the rock and phase compressibilities, and the densities in the fluxes
        and the well rates are treated as constant (so reservoir and surface
        volumes are not distinguished). Fidelity therefore requires
        $ c_t \\, Δp \\ll 1 $ -- and note that this is *not* a matter of
        choosing `ct` small: summing the
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

    wells: Any = None
    """The wells: a `Wells`, holding the flat, per-*completion* arrays --
    positions, rates, pressures, well indices -- that the model runs on.

    Assigning a list (or `dict`) of records -- one per well -- assembles one,
    which is the convenient way to configure them; the record format is
    documented in `TPFA_ResSim.wells.Wells.from_records`. Assigning `None`
    empties it. A `Wells` may also be given directly, in which case it is
    *bound* to this model, whereupon its completions snap onto the grid.

    The arrays remain writable throughout (`model.wells.rates = ...`), as an
    ensemble or optimisation loop requires, and there is no second,
    record-shaped copy of the configuration to fall out of step with them.
    """

    nComp = property(lambda self: self.wells.nComp)
    """Num. of well *completions*, i.e. the rows of every array the model
    indexes by them -- which is what it actually solves for, the equations
    being assembled per completion.
    Forwarded from `TPFA_ResSim.wells.Wells.nComp`.
    """

    def assemble_wells(
        self, S: np.ndarray | None, P: np.ndarray | None, k: int
    ) -> None:
        """Set up (for time `k`) the wells' contributions to the equations.

        The controls are those of `well_controls`, to which `S` and `P` (the
        state at the *start* of the step) are simply passed on.
        Rate-controlled wells enter the source/sink *field*, `_Q`, directly.
        BHP-controlled ones (ref `TPFA_ResSim.wells.Wells.bhp`) cannot: their
        rate is not yet known. They instead enter the pressure equations in
        `TPFA`, after which `realize_bhp` folds the resulting rate into `_Q`.
        """
        ctrl = self.well_controls(S, P, k)
        inds = self.xy2ind(*self.wells.xy.T)
        rates, p_bh = ctrl["rates"], ctrl["bhp"]
        is_bhp = np.isfinite(p_bh)
        assert np.isfinite(rates[~is_bhp]).all(), (
            "A rate-controlled well has a non-finite rate. Give it a number"
            " (`0` shuts it in), or put it on BHP control; ref `Wells.rates`."
        )

        # The well model's constant of proportionality, WI * λ_t.
        # NB: `nan` marks the rate-controlled wells, throughout.
        WI_lam = np.full(self.nComp, np.nan)
        if is_bhp.any():
            WI = self.wells.WI
            assert WI is not None and np.isfinite(WI[is_bhp]).all(), (
                "BHP control requires (finite) `Wells.WI`."
            )
            assert S is not None, "BHP control requires `S` (for λ_t)."
            Mw, Mo = self.RelPerm(S)
            WI_lam[is_bhp] = WI[is_bhp] * (Mw + Mo)[inds[is_bhp]]

        # Translate well conditions for cells.
        # NB: Dont use `Q[inds] += ...` since `inds` may contain dupes.
        self._Q, bhp_diag, bhp_rhs = np.zeros((3, self.Nxy))
        np.add.at(self._Q, inds[~is_bhp], rates[~is_bhp])
        np.add.at(bhp_diag, inds[is_bhp], WI_lam[is_bhp])
        np.add.at(bhp_rhs, inds[is_bhp], (WI_lam * p_bh)[is_bhp])
        rates[is_bhp] = np.nan  # only `realize_bhp` knows these
        self._wells_now: dict[str, np.ndarray] = dict(
            inds=inds, rates=rates, p_bh=p_bh,
            WI_lam=WI_lam, bhp_diag=bhp_diag, bhp_rhs=bhp_rhs,
        )  # fmt: skip

    def realize_bhp(self, P: np.ndarray) -> None:
        """Compute rates for BHP wells. Enter into `_Q` and `_wells_now["rates"]`.

        The rate, $ WI λ_t (p_\\mathrm{bh} - p_\\mathrm{cell}) $, is signed by
        nature: the flow direction is *emergent*, not declared (ref the
        `TPFA_ResSim.wells.Wells.bhp` warning).

        By construction of the linear system of `TPFA`, this leaves `_Q` equal
        to the *total* well flux, which is what keeps `storage_rate` -- and
        hence the transport step -- consistent with the pressure solution.
        """
        wls = self._wells_now
        WI_lam = wls["WI_lam"]  # `nan` marks the rate-controlled wells
        # Insert in cell source/sink field
        self._Q = self._Q + wls["bhp_rhs"] - wls["bhp_diag"] * P
        # Insert in per-well rates
        is_bhp = np.isfinite(WI_lam)
        wls["rates"][is_bhp] = (WI_lam * (wls["p_bh"] - P[wls["inds"]]))[is_bhp]

    def _record_actual_well_operation(
        self, S: np.ndarray, P: np.ndarray, k: int
    ) -> None:
        """Record `actual_rates`/`actual_bhp`. Warn about flow direction flip."""
        wls = self._wells_now
        if k:
            is_bhp = np.isfinite(wls["WI_lam"])
            flipped = is_bhp & (wls["rates"] * self.wells.actual_rates[:, k - 1] < 0)
            if flipped.any():
                warnings.warn(
                    f"BHP-controlled well(s) {np.flatnonzero(flipped).tolist()}"
                    f" reversed flow direction at step {k}"
                    " (an inflow injects water); ref `Wells.bhp`.",
                    stacklevel=2,
                )
        self.wells.actual_rates[:, k] = wls["rates"]
        self.wells.actual_bhp[:, k] = self.bhp(S, P, wls["rates"])

    def well_controls(self, S: np.ndarray | None, P: np.ndarray | None, k: int) -> dict:
        """Compute the wells' controls for time `k`: `dict(rates=..., bhp=...)`.

        Each is a `(nComp,)` array, read off the specifications --
        `TPFA_ResSim.wells.Wells.rates`, `TPFA_ResSim.wells.Wells.bhp` -- which
        are *open-loop*: fixed before the simulation begins. Overriding
        (patching/subclassing) this method is therefore how to do *feedback*
        control, the controls being free to depend on the state at the *start*
        of the step: the saturation `S` and the pressure `P`.
        The returned arrays are copies, so they may be modified in place.

        Most feedback concerns the rates alone -- e.g. shutting the wells upon
        water breakthrough at the producer:

        >>> class Shutter(ResSim):
        ...     def well_controls(self, S, P, k):
        ...         ctrl = super().well_controls(S, P, k)
        ...         if S is not None and S[self.xy2ind(1, 1)] > .5:
        ...             ctrl["rates"][:] = 0    # NB: all of them! See warning
        ...         return ctrl
        >>> model = Shutter(Lx=1, Ly=1, Nx=16, Ny=16,
        ...                 wells=Wells(xy=[[0, 0], [1, 1]], rates=[[1], [-1]]))
        >>> SS, PP = model.sim(.05, 20, model.swc*np.ones(model.Nxy), pbar=False)
        >>> int((model.wells.actual_rates[1] == 0).argmax())  # step of breakthrough
        16

        But the `bhp` is here too, and with it each well's *control mode*
        (`nan` => rate-controlled, ref `TPFA_ResSim.wells.Wells.bhp`) -- which
        is what an approximate mode *switch* requires. For example, rate
        control with a BHP limit -- the industrial default -- wherein a
        producer holds its rate only for as long as that does not draw it
        below some `p_min`:

        >>> class Limited(ResSim):
        ...     p_min = .5
        ...     def well_controls(self, S, P, k):
        ...         ctrl = super().well_controls(S, P, k)
        ...         if P is None:
        ...             return ctrl                    # nothing to switch on
        ...         p_bh = self.bhp(S, P, ctrl["rates"])
        ...         switch = p_bh < self.p_min         # the rate is unsustainable
        ...         ctrl["bhp"] = np.where(switch, self.p_min, np.nan)
        ...         return ctrl
        >>> model = Limited(Lx=1, Ly=1, Nx=16, Ny=16, ct=.1,
        ...                 wells=Wells(xy=[[.5, .5]], rates=[[-.25]]))
        >>> model.wells.WI = peaceman_WI(model, model.wells.xy, rw=1e-3)
        >>> SS, PP = model.sim(.02, 25, np.zeros(model.Nxy),
        ...                    P0=np.ones(model.Nxy), pbar=False)

        The well delivers its target rate until the limit binds, and declines
        thereafter -- at constant $ p_\\mathrm{bh} $, exponentially so
        (ref `examples/well_control.py`, which plots all three modes):

        >>> (-model.wells.actual_rates[0, [0, 5, 6, -1]]).round(3)
        array([0.25 , 0.25 , 0.182, 0.005])

        .. warning:: With `ct == 0` the rates must still sum to 0 at every step.

            Ref `TPFA_ResSim.wells.Wells.rates`. So shutting one well requires
            matching it on the other side -- as above -- else `time_stepper`
            complains that the "well rates do not sum to 0".
            Only with `ct > 0` (where storage
            absorbs the imbalance), or under BHP control
            (ref `TPFA_ResSim.wells.Wells.bhp`, where the well finds its own
            rate), may a well act alone.

        .. note:: The mode switch lags the solve by one step.

            It is decided from the previous step's pressure, whereas the well
            model itself is solved *simultaneously* with the new one (ref
            `TPFA_ResSim.wells.Wells.bhp`). So it is an approximation -- of the
            sort that a properly iterated switch would avoid -- and the limit is
            breached for the one step in which it comes to bind.
            Shorten `dt` to refine.

        .. note:: `S` and `P` may be `None`, so an override should tolerate that.

            They are `None` if the caller has none to offer -- as when
            `assemble_wells` is used merely to set up a plot.

        .. note:: Setting both controls for a well is not an error, just pointless.

            `assemble_wells` discards the rate of a BHP-controlled well -- it
            is `realize_bhp` that fills it in.
        """
        return dict(
            rates=self.wells.at_time("rates", 0.0, k),
            bhp=self.wells.at_time("bhp", np.nan, k),
        )

    def bhp(self, S: np.ndarray, P: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Bottom-hole pressures implied by the (signed) `rates`, via the well indices.

        I.e. the well model of `TPFA_ResSim.wells.Wells.WI`, solved for
        $ p_\\mathrm{bh} $:
        the rate's sign puts an injector above, a producer below, its cell
        pressure. `nan` wherever the well index is unset.

        `S` and `P` (both flat) should be the saturation and the pressure of the
        *same* `pressure_step`, i.e. `SS[k]` and `PP[k+1]` of `sim` -- which is
        what `actual_bhp` records, so prefer reading that.

        .. warning:: This is grid-independent, but inherits the model's premises.

            Unlike the cell pressure, it is (to the accuracy of the well model)
            independent of the grid resolution -- which is the whole point. But
            in particular $ λ_t $ is that of the well's *cell*, so an injector's
            injectivity is governed by the mobility of whatever the cell
            currently holds, rather than by that of the injectant.
        """
        if self.wells.WI is None:
            return np.full(self.nComp, np.nan)
        Mw, Mo = self.RelPerm(S)
        ii = self.xy2ind(*self.wells.xy.T)
        return P[ii] + rates / (self.wells.WI * (Mw + Mo)[ii])

    # Pres() -- listing 5
    def pressure_step(
        self,
        S: np.ndarray,
        P: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[np.ndarray, Fluxes]:
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
        [P, V] = self.TPFA(KM, P, dt)
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
    ) -> tuple[np.ndarray, Fluxes]:
        """Two-point flux-approximation (TPFA) of Darcy: $ -∇(K ∇u) = q $

        i.e. steady-state diffusion w/ nonlinear coefficient, $K$,
        if `ct == 0`. Otherwise (slightly compressible model) solve
        the backward-Euler step of $ φ c_t ∂u/∂t - ∇(K ∇u) = q $,
        which requires the previous pressure, `P`, and `dt`.

        After solving for pressure `P`, extract the fluxes `V`
        by finite differences.
        """
        # Compute transmissibilities by harmonic averaging.
        C = self.cdarcy
        L = 1 / K
        TX = np.zeros((self.Nx + 1, self.Ny))
        TY = np.zeros((self.Nx, self.Ny + 1))
        TX[1:-1, :] = C * 2 * self.hy / self.hx / (L[0, :-1, :] + L[0, 1:, :])
        TY[:, 1:-1] = C * 2 * self.hx / self.hy / (L[1, :, :-1] + L[1, :, 1:])

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
        elif not self._wells_now["bhp_diag"].any():
            # Pin the (o/w pure-Neumann & singular) problem.
            DiagVecs[2][0] += np.sum(self.K[:, 0, 0])  # ref article p. 13
        # Well model of the BHP-controlled wells
        DiagVecs[2] = DiagVecs[2] + self._wells_now["bhp_diag"]
        q = q + self._wells_now["bhp_rhs"]

        # Solve; compute A\q to update P
        A = self._spdiags(DiagVecs, DiagIndx)
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
        V = Fluxes(
            x=np.zeros((self.Nx + 1, self.Ny)),
            y=np.zeros((self.Nx, self.Ny + 1)),
        )
        V.x[1:-1, :] = (P2d[:-1, :] - P2d[1:, :]) * TX[1:-1, :]
        V.y[:, 1:-1] = (P2d[:, :-1] - P2d[:, 1:]) * TY[:, 1:-1]
        return P, V

    # GenA() -- listing 7
    def upwind_diff(self, V: Fluxes) -> sparse.dia_matrix:
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

    def storage_rate(self, V: Fluxes) -> np.ndarray:
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
    def estimate_1CFL(self, pv: np.ndarray, V: Fluxes, fi: np.ndarray) -> float:
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
    def saturation_step_upwind(self, S: np.ndarray, V: Fluxes, dt: float) -> np.ndarray:
        """Explicit upwind FV discretisation of conserv. of mass (water sat.)."""
        # fmt: off
        A  = self.upwind_diff(V)                 # FV discretized transport operator
        pv = self.h2 * self.por.ravel()          # Pore volume (per thickness)
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
        V: Fluxes,
        dt: float,
        nNewtonMax: int = 10,
        nTmax_log2: int = 10,
    ) -> np.ndarray:
        """Implicit FV discretisation of conserv. of mass (water sat.).

        .. warning:: The Newton iteration can converge to a spurious root.

            Far outside the $ c_t \\, Δp \\ll 1 $ regime (ref `ct`), it may
            converge -- silently -- to a root of the residual outside $[0, 1]$:
            the polynomial `RelPerm` extends smoothly beyond the unit interval,
            and the sub-`dt` halving only triggers on *non*-convergence.
            The explicit scheme
            (`saturation_step_upwind`), being monotone, stays within $[0, 1]$
            even for extreme `ct`.

        .. note:: This scheme rarely earns its keep.

            It is usually both slower and less accurate than
            `saturation_step_upwind` -- ref the "How to solve" section of the
            docs, and the branch `implicit-transport-scheme`.
        """
        # fmt: off
        A  = self.upwind_diff(V)                 # FV discretized transport operator
        pv = self.h2 * self.por.ravel()          # Pore volume (per thickness)
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

    def _validate(self):
        # Catch some common issues before they become mysterious/insidious
        # (e.g. mass imblance silently inserts deficit in SW corner).
        if self.ct == 0 and not self._wells_now["bhp_diag"].any():
            # Incompressible and no BHP control ⇒ no storage ⇒ src/sinks must balance.
            SA = np.abs(self._Q).sum()
            AS = abs(self._Q.sum())
            assert AS <= 1e-10 * SA, "well rates do not sum to 0"
        assert np.all((0 <= self.K) & np.isfinite(self.K))
        assert np.all((0 <= self.por) & (self.por <= 1))

    def time_stepper(self, dt: float, implicit: bool = False) -> Callable:
        """Get ODE solver (integrator) for model.

        Whatever time step `dt` is given, both schemes will use smaller steps internally.

        - `explicit`: computes sub-`dt` based on CFL esitmate.
        - `implicit`: reduces sub-`dt` until convergence is achieved.
        """

        def integrate(S, P, k):
            self.assemble_wells(S, P, k)
            self._validate()
            [P, V] = self.pressure_step(S, P, dt)
            self.realize_bhp(P)
            self._record_actual_well_operation(S, P, k)
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
        self.wells.actual_rates = np.zeros((self.nComp, nSteps))
        self.wells.actual_bhp = np.full((self.nComp, nSteps), np.nan)

        # Init
        SS[0] = S0
        if P0 is not None:
            PP[0] = P0

        # Recurse
        for k in kk:
            SS[k + 1], PP[k + 1] = step(SS[k], PP[k], k)

        return SS, PP
