""".. include:: README.md"""

import warnings
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
    __repr__ = NicePrint.__repr__
    __str__ = NicePrint.__str__

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
        if val is not None:
            # Well positions -- collocate at some node
            if key == "well_xy":
                val = np.array(val, float).reshape((-1, 2))
                for i, (x, y) in enumerate(val):
                    val[i] = self.ind2xy(self.xy2ind(x, y))
            # Well rates and/or pressures
            if key in ["well_rates", "well_bhp"]:
                val = np.array(val, float).reshape((self.nComp, -1))
            # Well indices
            if key == "well_WI":
                val = np.asarray(val, float)
                val = np.broadcast_to(val.ravel(), self.nComp).copy()
            # Completion-to-well map
            if key == "well_group":
                val = np.asarray(val, int).reshape(self.nComp)
            # Well configuration -- assemble the flat arrays that it specifies.
            # NB: this must precede the `super()` call below, since those arrays
            # each reset `wells` (also below) -- so it is stored last, and stands.
            if key == "wells":
                self._set_wells(val)
            # Permeabilities
            if key == "K":
                if np.isscalar(val):
                    val = np.full(self.shape, val, dtype=float)
                if val.size == self.size:
                    val = np.stack([val, val])  # both components
                val = val.reshape((2, *self.shape))
        # Set
        super().__setattr__(key, val)
        # A direct edit of the arrays outdates the records that produced them.
        # NB: `super()`, lest it be taken for a re-configuration (and recurse).
        if key.startswith("well_") and self.wells is not None:
            super().__setattr__("wells", None)

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

    nComp = property(lambda self: len(self.well_xy))
    """Num. of *completions*, i.e. rows of `well_xy`, which is what the model
    actually solves for. Several completions may compose a single well
    (ref `well_group`, `well_path`)."""

    nWell = property(
        lambda self: self.nComp if self.well_group is None
        else 1 + int(self.well_group.max())
    )
    """Num. of *wells*, i.e. groups of completions (ref `well_group`)."""

    well_xy: Any = None
    """Array of shape `(nComp, 2)` of x- and y-coords for the completions.

    Values should be betwen `0` and `Lx` or `Ly`.

    .. warning:: The wells get co-located with grid nodes, ref `xy2sub`.
        This is a design choice, not a mathematical necessity.
        An alternative would be to distribute them over nearby nodes.
    """
    well_rates: Any = None
    """Array of shape `(nComp, nTime)` -- or `(nComp, 1)` if constant-in-time.
    Ref `wells` for the convenient way to set this (and the other specs).

    **Signed**: a positive rate *injects* (water), a negative one *produces*
    (at the well cell's fractional flow). There is no other distinction
    between injectors and producers, anywhere in the model.

    .. note:: When `ct == 0` (and no well is BHP-controlled) it is asserted
        that the rates sum to 0 at each time index, otherwise the model
        would silently input the deficit from the SW corner.

    .. note:: Since the array is shared by all the wells, prefer `0` as (ignored)
        fill value where control is by `well_bhp`.
    """
    well_WI: Any = None
    """Well indices: `None`, or an array of shape `(nComp,)`, `nan` allowed.

    Compute $ WI $ with `peaceman_WI`, or set it directly
    (it need not come from any particular formula).
    A well whose entry is `nan` (or all of them, if `None`) has no well model:
    its `actual_bhp` is `nan`, and BHP control (`well_bhp`) unavailable.

    The well index, $ WI $, is the *sub-grid* well model: the constant of
    proportionality relating a well's (signed) flow rate to the pressure
    difference between the wellbore and the well's cell,
    $$ q = WI \\, λ_t \\, (p_\\mathrm{bh} - p_\\mathrm{cell}) \\,,$$
    with $ λ_t $ the total mobility (ref `RelPerm`) of the well's cell.
    The magnitude $ |p_\\mathrm{cell} - p_\\mathrm{bh}| $ is known as the
    "drawdown" (of a producer; "overpressure" for an injector). Re-arranging,
    $$ p_\\mathrm{bh} = p_\\mathrm{cell} + q / (WI \\, λ_t) \\,. $$

    NB: the drawdown cannot be replaced by plain cell pressure,
    because that is a cell average, which is highly sensitive to the
    chosen discretisation size around a point source/sink (an idealized,
    theoretical singularity).

    .. warning:: The drawdown is *not* a fixed offset that one could calibrate away once and for all.
        Being $ q / (WI \\, λ_t) $, it tracks the mobility -- which, in a waterflood,
        dips as the front arrives (by half, for equal viscosities). So the gap
        doubles at breakthrough: precisely when the well is most interesting.
    """
    well_bhp: Any = None
    """Bottom-hole pressures for the wells. `None`, or an array shaped like
    `well_rates`, i.e. `(nComp, nTime)` -- or `(nComp, 1)` if constant-in-time.

    A well whose entry is finite is **BHP-controlled** at that time. This is
    solved *simultaneously* with the pressure field (not lagged by a time step):
    `TPFA` puts $ WI λ_t $ on its diagonal and $ WI λ_t \\, p_\\mathrm{bh} $
    on its right-hand side.
    Only once $ p $ is known is the resulting rate folded into the source field,
    `_Q`, for the transport step -- ref `assemble_wells` and `realize_bhp`.

    Entries left as `nan` -- which is the default, for every well -- keep the
    well rate-controlled, per `well_rates`. So the two mechanisms can be mixed
    freely, both across wells and in time.

    Requires `well_WI` (finite, for the BHP-controlled wells). The realized
    rates are recorded in `actual_rates`, and the corresponding entries of
    `well_rates` are ignored (it may be left as `None`).

    .. note:: BHP control also *anchors the pressure*. With `ct == 0` the
        pressure equation is otherwise a pure-Neumann problem: solvable only up
        to a constant (which `TPFA` pins arbitrarily, ref article p. 13), and
        only if injection balances production. A single BHP well lifts both
        restrictions -- the level is then set by $ p_\\mathrm{bh} $, and the
        voidage by the well model.

    .. warning:: A BHP well's flow direction is *emergent*, not declared: it
        flows whichever way $ p_\\mathrm{bh} $ vs. its cell pressure dictates
        (ref `realize_bhp`) -- and, like any inflow, an inflow through a BHP
        well injects *water*. Since a reversal mid-`sim` may nonetheless be a
        surprise, a `UserWarning` is emitted when a BHP well's realized rate
        flips sign between steps of `sim`.
        Nor is there native switching of control modes:
        rate control with a BHP limit -- the industrial default -- would mean
        iterating each well's mode within each step. But `well_controls` sets
        the modes as well as the rates, and sees the previous step's pressure,
        so it can approximate the switch (as its docstring demonstrates) -- to
        within the lag thereby incurred.
    """

    well_group: Any = None
    """Which well each completion belongs to: `None`, or an int array of shape
    `(nComp,)` whose values index the wells, i.e. `well_names`.

    The model itself is indifferent to it: the equations are assembled per
    *completion* (ref `assemble_wells`), and the arrays -- `well_xy`,
    `well_rates`, `actual_rates`, ... -- are all indexed likewise. The grouping
    is what lets the *reporting* speak of wells nonetheless: ref
    `rates_by_well`, and the labels of `Plot2D.plt_field`.
    """
    well_names: Any = None
    """Names of wells (*not* completions): `None`, or a list of `nWell` strings."""

    wells: Any = None
    """Convenience well configuration: one record (`dict`) per well.

    Assigning it assembles the flat, per-*completion* arrays that the model
    runs on -- `well_xy`, `well_rates`, `well_bhp`, `well_WI` -- along with the
    grouping (`well_group`, `well_names`) by which the reporting recovers the
    wells. Those arrays remain writable, so whatever the records cannot express
    can still be set by hand afterwards.

    Each record may specify

    - `xy`: the well's position, `[x, y]` -- or positions,
      `[[x, y], ...]`, for a multi-completion well.
    - `path`: alternatively, a polyline, `[[x, y], ...]`, to be discretized
      into one completion per cell it traverses, ref `well_path`. Needs `rw`.
    - `rate`: the well's (signed, ref `well_rates`) rate: a scalar, or a
      schedule (an array over time). Apportioned among its completions in
      proportion to their well indices (uniformly, absent those).
    - `bhp`: alternatively (or, in time, additionally) the bottom-hole
      pressure, ref `well_bhp`. Scalar or schedule. Shared -- as a wellbore
      does -- by all of the well's completions.
    - `rw`, `skin`: the wellbore radius and skin, whence the well index, via
      `peaceman_WI`. Without them (or `WI`) the well has no well model.
    - `WI`: alternatively, the well index itself, given directly.
    - `name`: for the reporting. Defaults to the well's index.

    The concise cases stay concise -- a position and a rate is a well. A
    `dict` of records names them by its keys (as `name` does otherwise), and
    the constructor takes the same thing:

    >>> model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, wells={
    ...     "I1": dict(xy=[0, 0], rate=+1),
    ...     "P1": dict(xy=[1, 1], rate=-1, rw=1e-3),
    ... })
    >>> model.well_names
    ['I1', 'P1']
    >>> model.well_rates
    array([[ 1.],
           [-1.]])
    >>> model.well_WI.round(3)  # `P1` alone asked for a well model
    array([  nan, 2.498])

    A `path` becomes several completions of a single well, whose rate it
    shares out (ref `well_path`) and whose name they share:

    >>> model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10, wells=[
    ...     dict(name="I1", path=[[.05, .05], [.45, .05]], rate=+1, rw=1e-2),
    ...     dict(name="P1", xy=[.95, .95], rate=-1),
    ... ])
    >>> model.nComp, model.nWell
    (6, 2)
    >>> model.well_group
    array([0, 0, 0, 0, 0, 1])
    >>> model.well_rates.ravel().round(3)
    array([ 0.125,  0.25 ,  0.25 ,  0.25 ,  0.125, -1.   ])

    Schedules and control modes may be mixed freely across the wells: a
    constant is broadcast to the length of the longest schedule, while a
    spec that no well varies in time stays a singleton, which
    `well_controls` reads at any `k`. A BHP-controlled well leaves the
    (ignored) `0` in `well_rates`, and `nan` marks the rate-controlled ones
    in `well_bhp` -- the conventions that the specs, being shared arrays,
    call for (ref `well_rates`).

    .. note:: A well must be given a control (`rate` and/or `bhp`);
        `rate=0` shuts it in. This is deliberate: an uncontrolled well would
        silently be a shut one.

    .. note:: Assigning this is what applies it -- the normalization happening
        in `__setattr__`, as it does for `K` -- and it (re)configures the wells
        afresh, overwriting the flat arrays. Conversely, assigning any of
        *those* outdates the records that produced them, so `wells` is then
        reset to `None`: whatever it holds always describes the current wells.

    .. note:: The records are kept as given -- `well_names` being where the
        resolved names end up -- so this is also what the `repr` reports.
    """

    actual_rates: Any = None
    """The *realized* well rates: array of shape `(nComp, nSteps)`. Signed.

    Mostly used as a diagnostic in case of `well_bhp`. But even for
    rate-control it only coincides with `well_rates` up to broadcasting
    and assuming `well_controls` did not override it.
    """
    actual_bhp: Any = None
    """Like `actual_rates`, but the bottom-hole pressures (ref `bhp`).

    `nan` wherever the well index (`well_WI`) is unset.
    """

    def _set_wells(self, wells: Any) -> None:
        """Assemble the flat well arrays from the records of `wells`.

        Which is where the record format, and the conveniences it affords,
        are documented.
        """
        keys = ("name", "xy", "path", "rate", "bhp", "rw", "skin", "WI")
        if isinstance(wells, dict):
            wells = [dict(spec, name=name) for name, spec in wells.items()]
        names, xy, WI, group, rates, bhp = [], [], [], [], [], []
        specified: set = set()
        for i, well in enumerate(wells or []):
            spec = dict(well)
            if unknown := set(spec) - set(keys):
                raise TypeError(
                    f"Unknown key(s) in the spec of well {i}: {sorted(unknown)}."
                    f" Valid ones: {list(keys)}."
                )
            specified |= set(spec)
            name = str(spec.pop("name", i))
            rw, skin = spec.pop("rw", None), spec.pop("skin", 0.0)

            # Completions: their positions, and their well indices
            if (path := spec.pop("path", None)) is not None:
                assert "xy" not in spec, (
                    f"Well '{name}': give it `xy` or `path`, not both.")
                assert rw is not None, f"Well '{name}': a `path` requires `rw`."
                _xy, _WI, _ = self.well_path(path, rw, skin)
            else:
                assert "xy" in spec, f"Well '{name}': give it an `xy` (or a `path`)."
                _xy = np.array(spec.pop("xy"), float).reshape((-1, 2))
                _WI = (np.full(len(_xy), np.nan) if rw is None
                       else self.peaceman_WI(_xy, rw, skin))
            if (given := spec.pop("WI", None)) is not None:
                _WI = np.broadcast_to(np.asarray(given, float).ravel(), len(_xy)).copy()
            nc = len(_xy)

            # Apportion the rate by well index -- the standard, ref `well_path`
            alloc = np.full(nc, 1 / nc)
            if nc > 1 and np.isfinite(_WI).all() and _WI.sum() > 0:
                alloc = _WI / _WI.sum()

            # Controls. NB: the BHP is shared by the completions, the rate split
            rate, p_bh = spec.pop("rate", None), spec.pop("bhp", None)
            assert rate is not None or p_bh is not None, (
                f"Well '{name}' has no control: give it a `rate`"
                " (`0` shuts it in), or a `bhp`."
            )
            rate = 0.0 if rate is None else rate
            p_bh = np.nan if p_bh is None else p_bh
            rates.append(np.outer(alloc, np.ravel(rate)))
            bhp.append(np.outer(np.ones(nc), np.ravel(p_bh)))

            names.append(name)
            xy.append(_xy)
            WI.append(_WI)
            group.append(np.full(nc, i))

        if not names:
            for key in ["well_xy", "well_rates", "well_bhp",
                        "well_WI", "well_group", "well_names"]:
                setattr(self, key, None)
            return
        # NB: `well_xy` first -- it is what defines `nComp`, by which the
        # `__setattr__` normalization shapes the others.
        self.well_xy = np.vstack(xy)
        self.well_group = np.concatenate(group)
        self.well_names = names
        def stack(specs):
            """Stack the wells' `(nComp_i, nTime_i)` specs, widening the constants.

            NB: `_at_time` broadcasts a *wholly* singleton spec, but the array is
            shared, so a well held constant beside a scheduled one is widened here.
            """
            nTime = max(spec.shape[1] for spec in specs)
            assert all(spec.shape[1] in [1, nTime] for spec in specs), (
                "The wells' schedules must be of equal length (or constant):"
                f" got {sorted({spec.shape[1] for spec in specs})}."
            )
            return np.vstack([np.broadcast_to(s, (len(s), nTime)) for s in specs])

        # Leave a spec unset (`None`) if no well made use of it
        self.well_rates = stack(rates) if "rate" in specified else None
        self.well_bhp = stack(bhp) if "bhp" in specified else None
        WI = np.concatenate(WI)
        self.well_WI = WI if np.isfinite(WI).any() else None

    @property
    def rates_by_well(self) -> np.ndarray:
        """`actual_rates`, summed over each well's completions: `(nWell, nSteps)`."""
        group = np.arange(self.nComp) if self.well_group is None else self.well_group
        out = np.zeros((self.nWell, self.actual_rates.shape[1]))
        np.add.at(out, group, self.actual_rates)  # NB: `+=` would skip the dupes
        return out

    def assemble_wells(
        self, S: np.ndarray | None, P: np.ndarray | None, k: int
    ) -> None:
        """Set up (for time `k`) the wells' contributions to the equations.

        The controls are those of `well_controls`, to which `S` and `P` (the
        state at the *start* of the step) are simply passed on.
        Rate-controlled wells enter the source/sink *field*, `_Q`, directly.
        BHP-controlled ones (ref `well_bhp`) cannot: their rate is not yet known.
        They instead enter the pressure equations in `TPFA`, after which
        `realize_bhp` folds the resulting rate into `_Q`.
        """
        ctrl = self.well_controls(S, P, k)
        # `Any` coz ty cannot see that `DotDict` provides attribute access to keys
        wls: Any = DotDict(
            inds  = self.xy2ind(*self.well_xy.T),
            rates = ctrl["rates"],
            p_bh  = ctrl["bhp"],
        )  # fmt: off
        is_bhp = np.isfinite(wls.p_bh)
        assert np.isfinite(wls.rates[~is_bhp]).all(), (
            "A rate-controlled well has a non-finite rate. Give it a number"
            " (`0` shuts it in), or put it on BHP control; ref `well_rates`."
        )

        # The well model's constant of proportionality, WI * λ_t.
        # NB: `nan` marks the rate-controlled wells, throughout.
        wls.WI_lam = np.full(self.nComp, np.nan)
        if is_bhp.any():
            WI = self.well_WI
            assert WI is not None and np.isfinite(WI[is_bhp]).all(), (
                "BHP control requires (finite) `well_WI`."
            )
            assert S is not None, "BHP control requires `S` (for λ_t)."
            Mw, Mo = self.RelPerm(S)
            wls.WI_lam[is_bhp] = WI[is_bhp] * (Mw + Mo)[wls.inds[is_bhp]]

        # Translate well conditions for cells.
        # NB: Dont use `Q[inds] += ...` since `inds` may contain dupes.
        self._Q, wls.bhp_diag, wls.bhp_rhs = np.zeros((3, self.Nxy))
        np.add.at(self._Q, wls.inds[~is_bhp], wls.rates[~is_bhp])
        np.add.at(wls.bhp_diag, wls.inds[is_bhp], wls.WI_lam[is_bhp])
        np.add.at(wls.bhp_rhs, wls.inds[is_bhp], (wls.WI_lam * wls.p_bh)[is_bhp])
        wls.rates[is_bhp] = np.nan  # only `realize_bhp` knows these
        self._wells_now = wls

    def realize_bhp(self, P: np.ndarray) -> None:
        """Compute rates for BHP wells. Enter into `_Q` and `_wells_now.rates`.

        The rate, $ WI λ_t (p_\\mathrm{bh} - p_\\mathrm{cell}) $, is signed by
        nature: the flow direction is *emergent*, not declared (ref the
        `well_bhp` warning).

        By construction of the linear system of `TPFA`, this leaves `_Q` equal
        to the *total* well flux, which is what keeps `storage_rate` -- and
        hence the transport step -- consistent with the pressure solution.
        """
        wls = self._wells_now
        # Insert in cell source/sink field
        self._Q = self._Q + wls.bhp_rhs - wls.bhp_diag * P
        # Insert in per-well rates
        is_bhp = np.isfinite(wls.WI_lam)  # `nan` marks the rate-controlled wells
        wls.rates[is_bhp] = (wls.WI_lam * (wls.p_bh - P[wls.inds]))[is_bhp]

    def _record_actual_well_operation(self, S: np.ndarray, P: np.ndarray, k: int) -> None:
        """Record `actual_rates`/`actual_bhp`. Warn about flow direction flip."""
        wls = self._wells_now
        if k:
            is_bhp = np.isfinite(wls.WI_lam)
            flipped = is_bhp & (wls.rates * self.actual_rates[:, k - 1] < 0)
            if flipped.any():
                warnings.warn(
                    f"BHP-controlled well(s) {np.flatnonzero(flipped).tolist()}"
                    f" reversed flow direction at step {k}"
                    " (an inflow injects water); ref `well_bhp`.",
                    stacklevel=2,
                )
        self.actual_rates[:, k] = wls.rates
        self.actual_bhp[:, k] = self.bhp(S, P, wls.rates)

    def _at_time(self, spec: str, absent: float, k: int) -> np.ndarray:
        """Lookup the well `spec` (`"rates"`/`"bhp"`) at time `k`.

        Allows a constant-in-time (singleton) spec, and an unset (`None`) one
        (for which `0`/`nan` is returned for rate/bhp-controlled wells, respectively).
        Avoids broadcast (and potentially stale copies) to `(nComp, nSteps)`,
        which requires `nSteps`, i.e. `sim()`.
        """
        arr = getattr(self, f"well_{spec}")
        if arr is None:
            return np.full(self.nComp, absent)
        assert len(arr) == self.nComp, (
            f"`well_{spec}` has {len(arr)} rows, but there are"
            f" {self.nComp} completions (ref `well_xy`)."
        )
        # Copy, lest `well_controls` write into the spec itself
        return np.copy(arr[:, k if arr.shape[1] > 1 else 0])

    def well_controls(self, S: np.ndarray | None, P: np.ndarray | None, k: int) -> dict:
        """Compute the wells' controls for time `k`: `dict(rates=..., bhp=...)`.

        Each is a `(nComp,)` array, read off the specifications --
        `well_rates`, `well_bhp` -- which are *open-loop*: fixed before the
        simulation begins. Overriding (patching/subclassing) this method is
        therefore how to do *feedback* control, the controls being free to
        depend on the state at the *start* of the step: the saturation `S`
        and the pressure `P`.
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
        ...                 well_xy=[[0, 0], [1, 1]], well_rates=[[1], [-1]])
        >>> SS, PP = model.sim(.05, 20, model.swc*np.ones(model.Nxy), pbar=False)
        >>> int((model.actual_rates[1] == 0).argmax())  # step of breakthrough
        16

        But the `bhp` is here too, and with it each well's *control mode*
        (`nan` => rate-controlled, ref `well_bhp`) -- which is what an approximate
        mode *switch* requires. For example, rate control with a BHP limit --
        the industrial default -- wherein a producer holds its rate only for as
        long as that does not draw it below some `p_min`:

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
        ...                 well_xy=[[.5, .5]], well_rates=[[-.25]])
        >>> model.well_WI = model.peaceman_WI(model.well_xy, rw=1e-3)
        >>> SS, PP = model.sim(.02, 25, np.zeros(model.Nxy),
        ...                    P0=np.ones(model.Nxy), pbar=False)

        The well delivers its target rate until the limit binds, and declines
        thereafter -- at constant $ p_\\mathrm{bh} $, exponentially so
        (ref `examples/well_control.py`, which plots all three modes):

        >>> (-model.actual_rates[0, [0, 5, 6, -1]]).round(3)
        array([0.25 , 0.25 , 0.182, 0.005])

        .. warning:: With `ct == 0` the rates must still sum to 0 at every step
            (ref `well_rates`), so shutting one well requires matching it on the
            other side -- as above -- else `time_stepper` complains that the
            "well rates do not sum to 0". Only with `ct > 0` (where storage
            absorbs the imbalance), or under BHP control (ref `well_bhp`, where
            the well finds its own rate), may a well act alone.

        .. note:: The mode switch is decided from the previous step's pressure,
            whereas the well model itself is solved *simultaneously* with the
            new one (ref `well_bhp`). So it is an approximation -- of the sort
            that a properly iterated switch would avoid -- and the limit is
            breached for the one step in which it comes to bind.
            Shorten `dt` to refine.

        .. note:: `S` and `P` are `None` if the caller has none to offer (as
            when `assemble_wells` is used merely to set up a plot), so an
            override should tolerate that.

        .. note:: `assemble_wells` discards the rate of a BHP-controlled well
            (it is `realize_bhp` that fills it in), so setting both controls for
            the same well is not an error, merely pointless.
        """
        return dict(
            rates=self._at_time("rates", 0.0, k),
            bhp=self._at_time("bhp", np.nan, k),
        )

    def peaceman_WI(self, xy: Any, rw: float, skin: float = 0.0) -> np.ndarray:
        """Peaceman's well index for wells at `xy`, of radius `rw`.

        Called for a well of `wells` given an `rw`, which is the convenient
        way to use it.

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

    def well_path(self, vertices: Any, rw: float, skin: float = 0.0) -> tuple:
        """Discretize a well *path* (a polyline) into 1 weighted completion per traversed cell.

        Called for a well of `wells` given a `path`, which is the convenient
        way to use it: the three returned arrays then need not be assembled
        (with those of the other wells) by hand.

        Returns `(xy, WI, alloc)`:

        - `xy`: centres of the cells that the path traverses -- i.e. a value for
          `well_xy`. Several completions act as a single well simply by
          being several wells: `assemble_wells` superimposes them.
        - `WI`: their well indices, i.e. a value for `well_WI`. Each is
          `peaceman_WI` for its cell, scaled by the fraction of that cell which
          the path actually traverses (so a cell merely clipped by the path
          contributes proportionally less).
        - `alloc`: `WI / WI.sum()`, for apportioning the rate among its completions:
          `well_rates = rate * alloc[:, None]` (the rate signed as usual).
          This is the standard (static) allocation -- proportional to the well index,
          hence to both the contacted length and the local permeability.

        >>> model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10)
        >>> xy, WI, alloc = model.well_path([[.05, .05], [.45, .05]], rw=1e-2)
        >>> xy.T[0]  # the traversed cells, in x
        array([0.05, 0.15, 0.25, 0.35, 0.45])
        >>> alloc  # the end cells, entered mid-way, get half the rate
        array([0.125, 0.25 , 0.25 , 0.25 , 0.125])

        .. note:: Under BHP control this `alloc` is exact (assuming 0 gravity and friction):
            the completions simply share a `p_bh`.
            Under *rate* control, `alloc` is an approximation unless the cell pressures are equal.
            Solving for it would make $ p_\\mathrm{bh} $ an extra
            unknown, i.e. a bordered linear system -- which the 5-diagonal
            assembly of `TPFA` (Listing 1) is not set up for. Use
            `well_controls` to reallocate per step, if it matters.

        .. warning:: The completions are treated as independent *vertical* wells,
            which seems reasonable in a 2D areal model. Thus they count towards
            `nComp`, not `nWell` -- ref `well_group`, which `wells` sets for you.
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

    def bhp(self, S: np.ndarray, P: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Bottom-hole pressures implied by the (signed) `rates`, via the well indices.

        I.e. the well model of `well_WI`, solved for $ p_\\mathrm{bh} $:
        the rate's sign puts an injector above, a producer below, its cell
        pressure. `nan` wherever the well index is unset.

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
        if self.well_WI is None:
            return np.full(self.nComp, np.nan)
        Mw, Mo = self.RelPerm(S)
        ii = self.xy2ind(*self.well_xy.T)
        return P[ii] + rates / (self.well_WI * (Mw + Mo)[ii])

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
        elif not self._wells_now.bhp_diag.any():
            # Pin the (o/w pure-Neumann & singular) problem.
            DiagVecs[2][0] += np.sum(self.K[:, 0, 0])  # ref article p. 13
        # Well model of the BHP-controlled wells
        DiagVecs[2] = DiagVecs[2] + self._wells_now.bhp_diag
        q = q + self._wells_now.bhp_rhs

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
            self.assemble_wells(S, P, k)

            # Catch some common issues before they become mysterious/insidious
            # (e.g. mass imblance silently inserts deficit in SW corner).
            if self.ct == 0 and not self._wells_now.bhp_diag.any():
                # Incompressible and no BHP control ⇒ no storage ⇒ src/sinks must balance.
                assert np.isclose(self._Q.sum(), 0), "well rates do not sum to 0"
            assert np.all((0 <= self.K) & np.isfinite(self.K))
            assert np.all((0 <= self.por) & (self.por <= 1))

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
        self.actual_rates = np.zeros((self.nComp, nSteps))
        self.actual_bhp = np.full((self.nComp, nSteps), np.nan)

        # Init
        SS[0] = S0
        if P0 is not None:
            PP[0] = P0

        # Recurse
        for k in kk:
            SS[k + 1], PP[k + 1] = step(SS[k], PP[k], k)

        return SS, PP
