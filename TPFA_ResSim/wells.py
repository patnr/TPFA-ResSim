"""The wells: their configuration, their bookkeeping, and the well model.

Everything here depends on the wells alone, on the *geometry* they sit in
(`TPFA_ResSim.grid.Grid2D`), and -- for the well index -- on the permeability
`K`. What couples the wells to the *fluids* (the mobilities of
`TPFA_ResSim.ResSim.RelPerm`) or to the linear system (the source field, the
BHP contributions) stays with the simulator, which reads the arrays assembled
here.

The unit of account is the **completion**, not the well: a well may span
several cells (ref `well_path`), and it is the completions that the model
solves for. Hence `Wells.nComp` counts the rows of every array here, whereas
`Wells.nWell` counts the wells that `Wells.group` says they compose --
a distinction that serves the *reporting* alone (`Wells.rates_by_well`, the
plot labels); the physics never groups.


.. note:: Spreading a well over its neighbouring cells was implemented, then omitted.

    Snapped to a cell centre, a well has its every effect a staircase function
    of its coordinates -- constant within a cell -- which leaves an
    optimisation over well *positions* with no gradient to work with until the
    well crosses into the next cell (as in the EnOpt of
    [HistoryMatching](https://github.com/patnr/HistoryMatching)). Distributing
    it over the 4 surrounding cells by bilinear weights -- a *mollified* rather
    than a rounded delta -- fixes that, and is not merely cosmetic: the
    position dependence so obtained tracks that of a 3-times-refined grid to
    within that grid's own discretization spread, i.e. some 5 times closer than
    rounding manages. It does require the well index to be corrected for the
    cells dividing the load, Peaceman's equivalent radius being derived for the
    whole source in one cell: a well divided 4 ways under-reports its drawdown
    by 23%, non-convergently, unless the weighted geometric mean of the
    intercell distances is substituted for $r_e$ (which recovers 0.3%).

    It was nonetheless judged not to earn its complexity -- a stencil threaded
    through the well assembly, the well model and the plotting, for a
    convenience of the optimiser rather than a fidelity of the simulator. The
    work is preserved on the branch `well-spread` (`ResSim.spread_wells`,
    `Grid2D.xy2stencil`, `wells._share_WI`, and `tests/test_spread.py`, which
    pins each of the measurements quoted above).

## Theory

A well is either an **injector** or a **producer** -- the terminal states of the
sources and sinks, $q$, of the governing equations.
The two are not distinct objects here: a well is an injector or a producer
merely by the *sign* of its rate (ref `TPFA_ResSim.wells.Wells.rates`), and
under BHP control not even by that, the direction being left to the pressures
(ref `TPFA_ResSim.wells.Wells.bhp`).
Its **completion** is the equipment that connects the **wellbore** to the rock,
whose interface is the **sandface**; it may be *open hole*, or cased and
**perforated**. A well may have several completions, e.g. one per layer,
or (here) one per grid cell traversed by the well path -- which the model
assembles individually, grouping them back into wells only for the reporting
(ref `TPFA_ResSim.ResSim.wells`).

Because a wellbore (radius $r_w \\sim 0.1$ m) is orders of magnitude smaller than a
grid block, its pressure is not resolved by the grid: the radial solution
$p \\sim \\ln r$ spends most of its variation within the well's own cell.
The **bottom-hole pressure** (BHP), $p_\\mathrm{bh}$, is the pressure at the
sandface, which is thus a *sub-grid* quantity;
measured at the surface instead, it is the *tubing head pressure* (THP), the
difference being hydrostatic and friction losses in the tubing -- neither of which
a 2D areal model has, which is why BHP is where this model stops.
The **drawdown** is the pressure difference driving the flow,
$p_\\mathrm{cell} - p_\\mathrm{bh}$: positive for a producer (whence its name),
while for an injector it is negative, being an *overpressure*.

The **productivity index** (PI) is the resulting constant of proportionality,
$q = \\mathrm{PI} \\cdot \\Delta p$ (the *injectivity index* for injectors),
familiar from well testing. Its counterpart in a simulator is the
**well index** (WI), which is the same relation with the fluid factored out,
$q = \\mathrm{WI} \\, \\lambda_t \\, \\Delta p$,
so that $\\mathrm{WI}$ depends only on geometry and rock,
and the mobility $\\lambda_t$ carries the (time-varying) fluid dependence.
It is the well's analogue of the transmissibility $t_{ij}$ of eqn. (10).
Two things enter it, beyond $r_w$ and $\\mathbf{K}$:

- The **skin**, $S$, is a dimensionless lumping of all *near-wellbore* effects that
  the model does not resolve. It is *positive* for damage (drilling mud invasion,
  fines migration, scale) and *negative* for stimulation (acidizing, or hydraulic
  **fracturing**), and enters additively to a logarithm, so a skin of $5$ is a lot.
- The **equivalent radius**, $r_e$, is the purely *numerical* ingredient: the radius
  at which the analytic radial pressure equals the numerical *cell* pressure,
  $r_e \\approx 0.2 h$ for the 5-point stencil of TPFA. Beware that the same symbol
  is used, in well testing, for the (physical) *drainage radius*.

Ref `TPFA_ResSim.wells.peaceman_WI` for the formula combining these.

A well is **controlled** either by prescribing its rate, or its BHP,
the other then being an outcome (ref `TPFA_ResSim.wells.Wells.bhp`).
Reality is closer to the latter -- one sets a pump speed or a **choke** opening,
and the reservoir decides the rate -- but the *rate* is what is usually planned for.
Field practice is therefore rate control subject to a BHP *constraint*
(from the fracture pressure of an injector, or the lift capacity, or the bubble
point, of a producer), **switching** control mode whenever the constraint binds.
A well producing at a rate too low to be worthwhile is **shut in**;
if it cannot flow unaided it needs **artificial lift** (gas lift, or a downhole pump).

The **water cut** of a producer is the water fraction of what it produces,
i.e. the $f(s)$ of its cell; **breakthrough** is when the injected water first
arrives, after which the water cut climbs and the well eventually becomes uneconomic.
How much of the oil the flood has contacted by then is the **sweep efficiency**,
which is governed by the **mobility ratio**, $M = \\lambda_w / \\lambda_o$
evaluated behind and ahead of the front. $M > 1$ is *unfavourable*:
the (less viscous) water outruns the oil in **viscous fingering**,
and even more so along the high-permeability *channels* -- the motivation for
polymer injection, which fixes $M$ by thickening the water.
The corresponding vertical phenomenon (which a 2D areal model cannot see)
is **coning**, of water up, or gas down, into the completion.

Wells are drilled in repeated **patterns**, of which the *five-spot* (a producer
at the centre of four injectors, or vice-versa if *inverted*) is the classic;
by symmetry it suffices to simulate the *quarter five-spot*, as in the examples here.
They need not be vertical: *deviated*, *horizontal* and *multilateral* wells
contact more rock per well, at the price of an **allocation** problem,
namely how the total rate distributes itself among the completions
(ref `TPFA_ResSim.wells.well_path`).
Later interventions to restore or improve a well are **workovers**,
and drilling extra wells between the existing ones is **infill drilling**.

"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from TPFA_ResSim._repr import AlignedRepr

if TYPE_CHECKING:
    from TPFA_ResSim import ResSim


def peaceman_WI(model: "ResSim", xy: Any, rw: float, skin: float = 0.0) -> np.ndarray:
    """Peaceman's well index for wells at `xy`, of radius `rw`, in `model`.

    Applied for you to a well of `Wells.from_records` given an `rw`,
    which is the convenient way to use it.

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
    the 5-point stencil that `TPFA_ResSim.ResSim.TPFA` assembles, and this model
    reproduces it (`tests/test_wells.py` recovers $ r_e / h → 0.198 $ from the
    simulated drawdown, and thereby the analytic, radial well pressure to
    within 0.2%, on grids from 16² to 64²).

    >>> from TPFA_ResSim import ResSim
    >>> model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32)
    >>> peaceman_WI(model, [[.5, .5]], rw=1e-3).round(4)
    array([3.4476])

    .. note:: `model` is used for its grid and its `K` alone.

        Which is why this is a free function, not a method: the well index is a
        property of a *location*, not of the well configuration, and it is
        evaluated once, when asked for -- so a later edit of `K` does not
        retroactively change a `Wells.WI` computed from it.

    .. note:: $ WI \\, λ_t \\, Δp $ comes out as a rate *per unit thickness*.

        Ref. `TPFA_ResSim.ResSim.cdarcy`.

    .. note:: `rw` must be given in the same length unit as `Lx`.
    """
    xy = np.asarray(xy, float).reshape((-1, 2))
    ix, iy = model.xy2sub(*xy.T)
    kx, ky = model.K[0][ix, iy], model.K[1][ix, iy]
    # fmt: off
    a, b = np.sqrt(ky/kx), np.sqrt(kx/ky)
    r_e  = .28 * np.sqrt(a*model.hx**2 + b*model.hy**2) / (a**.5 + b**.5)
    return model.cdarcy * 2*np.pi*np.sqrt(kx*ky) / (np.log(r_e/rw) + skin)
    # fmt: on


def well_path(model: "ResSim", vertices: Any, rw: float, skin: float = 0.0) -> tuple:
    """Discretize a well *path* (a polyline): 1 weighted completion per cell.

    Applied for you to a well of `Wells.from_records` given a `path`, which is
    the convenient way to use it: the three returned arrays then need not be
    assembled (with those of the other wells) by hand.

    Returns `(xy, WI, alloc)`:

    - `xy`: centres of the cells that the path traverses -- i.e. a value for
      `Wells.xy`. Several completions act as a single well simply by
      being several wells: `TPFA_ResSim.ResSim.assemble_wells` superimposes them.
    - `WI`: their well indices, i.e. a value for `Wells.WI`. Each is
      `peaceman_WI` for its cell, scaled by the fraction of
      that cell which the path actually traverses (so a cell merely clipped
      by the path contributes proportionally less).
    - `alloc`: `WI / WI.sum()`, for apportioning the rate among its completions:
      `rates = rate * alloc[:, None]` (the rate signed as usual).
      This is the standard (static) allocation -- proportional to the well index,
      hence to both the contacted length and the local permeability.

    >>> from TPFA_ResSim import ResSim
    >>> model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10)
    >>> xy, WI, alloc = well_path(model, [[.05, .05], [.45, .05]], rw=1e-2)
    >>> xy.T[0]  # the traversed cells, in x
    array([0.05, 0.15, 0.25, 0.35, 0.45])
    >>> alloc  # the end cells, entered mid-way, get half the rate
    array([0.125, 0.25 , 0.25 , 0.25 , 0.125])

    .. note:: `alloc` is exact under BHP control, approximate under rate control.

        Under BHP control (assuming 0 gravity and friction) the completions
        simply share a `p_bh`. Under *rate* control, the allocation holds only
        if the cell pressures are equal. Solving for it would make
        $ p_\\mathrm{bh} $ an extra
        unknown, i.e. a bordered linear system -- which the 5-diagonal
        assembly of `TPFA_ResSim.ResSim.TPFA` (Listing 1) is not set up for. Use
        `TPFA_ResSim.ResSim.well_controls` to reallocate per step, if it matters.

    .. warning:: The completions are treated as independent *vertical* wells.

        Which seems reasonable in a 2D areal model. Thus they count towards
        `Wells.nComp`, not `Wells.nWell` -- ref `Wells.group`, which
        `Wells.from_records` sets for you.
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
        ts = model._crossings(p0, d)
        mids = p0 + np.outer((ts[:-1] + ts[1:]) / 2, d)
        for mid, dt in zip(mids, np.diff(ts)):
            sub = tuple(int(i) for i in model.xy2sub(*mid))
            lengths[sub] = lengths.get(sub, 0.0) + L * dt
    # Discard the slivers left by corner crossings
    total = sum(lengths.values())
    lengths = {k: v for k, v in lengths.items() if v > 1e-9 * total}

    subs = np.array(list(lengths))
    xy = model.sub2xy(*subs.T).T
    # Scale each WI by how much of its cell the path traverses, relative
    # to the cell size -- so an axis-aligned full crossing scores exactly 1
    # (and a diagonal one √2, it contacting that much more rock).
    frac = np.array(list(lengths.values())) / np.sqrt(model.h2)
    WI = frac * peaceman_WI(model, xy, rw, skin)
    return xy, WI, WI / WI.sum()


@dataclass
class Wells(AlignedRepr):
    """The wells of a `TPFA_ResSim.ResSim`: the flat, per-completion arrays.

    These arrays *are* the configuration -- there is no second, record-shaped
    copy of it to fall out of step -- and they are meant to be written to, as
    an ensemble or optimisation loop does:

    >>> from TPFA_ResSim import ResSim
    >>> model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16,
    ...                wells=Wells(xy=[[0, 0], [1, 1]], rates=[[1], [-1]]))
    >>> model.wells.rates = [[2], [-2]]
    >>> model.wells.nComp
    2

    Assigning one normalizes it (`__setattr__`): the positions get snapped onto
    the grid nodes, the schedules reshaped to `(nComp, nTime)`, and so on.
    A `Wells` may also be built before it has a grid to snap to -- as above,
    where it is the assignment to the model that binds (and snaps) it.

    Ref `from_records` for the convenient, well-shaped way to configure them,
    and `TPFA_ResSim.ResSim.wells` for the attribute that holds them.
    """

    # Dont use dataclass repr
    __repr__ = AlignedRepr.__repr__

    # NB: the array attributes are typed `Any` since `__setattr__` normalizes
    # whatever array-like (nested lists, scalars) is assigned to them.
    xy: Any = None
    """Array of shape `(nComp, 2)` of x- and y-coords for the completions.

    Values should be betwen `0` and `Lx` or `Ly`. Empty, shape `(0, 2)`, if
    there are no wells.

    .. warning:: The wells get co-located with grid nodes, ref `xy2sub`.

        This is a design choice, not a mathematical necessity.
        An alternative would be to distribute them over nearby nodes.
    """
    rates: Any = None
    """Array of shape `(nComp, nTime)` -- or `(nComp, 1)` if constant-in-time.
    Ref `from_records` for the convenient way to set this (and the other specs).

    **Signed**: a positive rate *injects* (water), a negative one *produces*
    (at the well cell's fractional flow). There is no other distinction
    between injectors and producers, anywhere in the model.
    An **areal** rate, i.e. volumetric *per unit thickness* -- ref
    `TPFA_ResSim.ResSim.cdarcy`.

    .. note:: With `ct == 0` the rates must sum to 0 at each time index.

        This is asserted (unless a well is BHP-controlled), otherwise the model
        would silently input the deficit from the SW corner.

    .. note:: Prefer `0` as the (ignored) fill value where control is by `bhp`.

        The array being shared by all the wells, an entry must be given there
        regardless.
    """
    bhp: Any = None
    """Bottom-hole pressures for the wells. `None`, or an array shaped like
    `rates`, i.e. `(nComp, nTime)` -- or `(nComp, 1)` if constant-in-time.

    A well whose entry is finite is **BHP-controlled** at that time. This is
    solved *simultaneously* with the pressure field (not lagged by a time step):
    `TPFA_ResSim.ResSim.TPFA` puts $ WI λ_t $ on its diagonal and
    $ WI λ_t \\, p_\\mathrm{bh} $ on its right-hand side.
    Only once $ p $ is known is the resulting rate folded into the source field,
    `_Q`, for the transport step -- ref `TPFA_ResSim.ResSim.assemble_wells`
    and `TPFA_ResSim.ResSim.realize_bhp`.

    Entries left as `nan` -- which is the default, for every well -- keep the
    well rate-controlled, per `rates`. So the two mechanisms can be mixed
    freely, both across wells and in time.

    Requires `WI` (finite, for the BHP-controlled wells). The realized
    rates are recorded in `actual_rates`, and the corresponding entries of
    `rates` are ignored (it may be left as `None`).

    .. note:: BHP control also *anchors the pressure*.

        With `ct == 0` the pressure equation is otherwise a pure-Neumann
        problem: solvable only up to a constant (which
        `TPFA_ResSim.ResSim.TPFA` pins arbitrarily, ref article p. 13), and
        only if injection balances production. A single BHP
        well lifts both restrictions -- the level is then set by
        $ p_\\mathrm{bh} $, and the voidage by the well model.

    .. warning:: A BHP well's flow direction is *emergent*, not declared.

        It flows whichever way $ p_\\mathrm{bh} $ vs. its cell pressure dictates
        (ref `TPFA_ResSim.ResSim.realize_bhp`) -- and, like any inflow, an
        inflow through a BHP well injects *water*. Since a reversal mid-`sim`
        may nonetheless be a surprise, a `UserWarning` is emitted when a BHP
        well's realized rate flips sign between steps of `sim`.
        Nor is there native switching of control modes:
        rate control with a BHP limit -- the industrial default -- would mean
        iterating each well's mode within each step. But
        `TPFA_ResSim.ResSim.well_controls` sets the modes as well as the rates,
        and sees the previous step's pressure, so it can approximate the switch
        (as its docstring demonstrates) -- to within the lag thereby incurred.
    """
    WI: Any = None
    """Well indices: `None`, or an array of shape `(nComp,)`, `nan` allowed.

    Compute $ WI $ with `peaceman_WI`, or set it directly
    (it need not come from any particular formula).
    A well whose entry is `nan` (or all of them, if `None`) has no well model:
    its `actual_bhp` is `nan`, and BHP control (`bhp`) unavailable.

    The well index, $ WI $, is the *sub-grid* well model: the constant of
    proportionality relating a well's (signed) flow rate to the pressure
    difference between the wellbore and the well's cell,
    $$ q = WI \\, λ_t \\, (p_\\mathrm{bh} - p_\\mathrm{cell}) \\,,$$
    with $ λ_t $ the total mobility (ref `TPFA_ResSim.ResSim.RelPerm`) of the
    well's cell. The magnitude $ |p_\\mathrm{cell} - p_\\mathrm{bh}| $ is known
    as the "drawdown" (of a producer; "overpressure" for an injector).
    Re-arranging,
    $$ p_\\mathrm{bh} = p_\\mathrm{cell} + q / (WI \\, λ_t) \\,. $$

    NB: the drawdown cannot be replaced by plain cell pressure,
    because that is a cell average, which is highly sensitive to the
    chosen discretisation size around a point source/sink (an idealized,
    theoretical singularity).

    .. warning:: The drawdown is *not* a fixed offset.

        It is not to be calibrated away once and for all: being
        $ q / (WI \\, λ_t) $, it tracks the mobility --
        which, in a waterflood, dips as the front arrives (by half, for equal
        viscosities). So the gap doubles at breakthrough: precisely when the
        well is most interesting.
    """
    group: Any = None
    """Which well each completion belongs to: `None`, or an int array of shape
    `(nComp,)` whose values index the wells, i.e. `names`.

    The model itself is indifferent to it: the equations are assembled per
    *completion* (ref `TPFA_ResSim.ResSim.assemble_wells`), and the arrays --
    `xy`, `rates`, `actual_rates`, ... -- are all indexed likewise. The grouping
    is what lets the *reporting* speak of wells nonetheless: ref
    `rates_by_well`, and the labels of `TPFA_ResSim.plotting.Plot2D.plt_field`.
    """
    names: Any = None
    """Names of wells (*not* completions): `None`, or a list of `nWell` strings."""

    actual_rates: Any = None
    """The *realized* well rates: array of shape `(nComp, nSteps)`. Signed.

    Mostly used as a diagnostic in case of `bhp`. But even for
    rate-control it only coincides with `rates` up to broadcasting
    and assuming `TPFA_ResSim.ResSim.well_controls` did not override it.
    """
    actual_bhp: Any = None
    """Like `actual_rates`, but the bottom-hole pressures.

    `nan` wherever the well index (`WI`) is unset.
    """

    _grid: Any = None
    """The model these wells are in -- set when assigned to it. Used only for
    the grid geometry (which is why it is typed as such, ref `_bind`)."""

    def __setattr__(self, key: str, val: Any) -> None:
        # NB: the single normalization layer for the wells -- `from_records`
        # routes its assignments through it rather than writing past it.
        if key == "xy":
            # Completion positions -- collocate at some node
            val = (
                np.zeros((0, 2))
                if val is None
                else np.array(val, float).reshape((-1, 2))
            )
            if self._grid is not None:
                for i, (x, y) in enumerate(val):
                    val[i] = self._grid.ind2xy(self._grid.xy2ind(x, y))
        elif val is not None:
            # Rates and/or pressures
            if key in ["rates", "bhp"]:
                val = np.array(val, float).reshape((self.nComp, -1))
            # Well indices
            elif key == "WI":
                val = np.broadcast_to(np.asarray(val, float).ravel(), self.nComp).copy()
            # Completion-to-well map
            elif key == "group":
                val = np.asarray(val, int).reshape(self.nComp)
        super().__setattr__(key, val)

    def _bind(self, grid: Any) -> None:
        """Attach to `grid` (a `TPFA_ResSim.grid.Grid2D`, i.e. the model).

        Whereupon the completions snap onto its nodes -- which an unbound
        `Wells`, having no grid to snap to, could not do.
        """
        self._grid = grid
        self.xy = self.xy  # re-normalize, now that there is a grid

    nComp = property(lambda self: len(self.xy))
    """Num. of *completions*, i.e. rows of `xy`, which is what the model
    actually solves for. Several completions may compose a single well
    (ref `group`, `well_path`)."""

    nWell = property(
        lambda self: self.nComp if self.group is None else 1 + int(self.group.max())
    )
    """Num. of *wells*, i.e. groups of completions (ref `group`)."""

    @property
    def rates_by_well(self) -> np.ndarray:
        """`actual_rates`, summed over each well's completions: `(nWell, nSteps)`."""
        group = np.arange(self.nComp) if self.group is None else self.group
        out = np.zeros((self.nWell, self.actual_rates.shape[1]))
        np.add.at(out, group, self.actual_rates)  # NB: `+=` would skip the dupes
        return out

    @property
    def signs(self) -> np.ndarray:
        """The sign (`+1` inject, `-1` produce, `0` unknown) of each well's rate.

        Read off the *spec*, `rates`, summed over time (`nan` entries --
        which a BHP-controlled well may well have -- being skipped). Wells left
        undecided by it, i.e. those with no spec or a vanishing one (as under
        pure BHP control), fall back on the `actual_rates` of the latest `sim`,
        if there has been one. Only the truly undecided are then `0`.
        """
        sgn = np.zeros(self.nComp, int)
        for rates in [self.rates, self.actual_rates]:
            if rates is not None:
                q = np.nansum(rates, axis=1)
                sgn = np.where(sgn, sgn, (q > 0).astype(int) - (q < 0))
        return sgn

    def at_time(self, spec: str, absent: float, k: int) -> np.ndarray:
        """Lookup the `spec` (`"rates"`/`"bhp"`) at time `k`.

        Allows a constant-in-time (singleton) spec, and an unset (`None`) one
        (for which `absent`, i.e. `0`/`nan` for rate/bhp-controlled wells
        respectively, is returned).
        Avoids broadcast (and potentially stale copies) to `(nComp, nSteps)`,
        which requires `nSteps`, i.e. `sim()`.
        """
        arr = getattr(self, spec)
        if arr is None:
            return np.full(self.nComp, absent)
        assert len(arr) == self.nComp, (
            f"`wells.{spec}` has {len(arr)} rows, but there are"
            f" {self.nComp} completions (ref `Wells.xy`)."
        )
        # Copy, lest `well_controls` write into the spec itself
        return np.copy(arr[:, k if arr.shape[1] > 1 else 0])

    @classmethod
    def from_records(cls, model: "ResSim", wells: Any) -> "Wells":
        """Assemble the flat, per-completion arrays from one record (`dict`) per well.

        This is the convenient way to configure the wells, and assigning the
        records to `TPFA_ResSim.ResSim.wells` is what applies it. Each record
        may specify

        - `xy`: the well's position, `[x, y]` -- or positions,
          `[[x, y], ...]`, for a multi-completion well.
        - `path`: alternatively, a polyline, `[[x, y], ...]`, to be discretized
          into one completion per cell it traverses, ref
          `well_path`. Needs `rw`.
        - `rate`: the well's (signed, ref `Wells.rates`) rate: a scalar, or a
          schedule (an array over time). Apportioned among its completions in
          proportion to their well indices (uniformly, absent those).
        - `bhp`: alternatively (or, in time, additionally) the bottom-hole
          pressure, ref `Wells.bhp`. Scalar or schedule. Shared -- as a wellbore
          does -- by all of the well's completions.
        - `rw`, `skin`: the wellbore radius and skin, whence the well index, via
          `peaceman_WI`. Without them (or `WI`) the well has no well model.
        - `WI`: alternatively, the well index itself, given directly.
        - `name`: for the reporting. Defaults to the well's index.

        The concise cases stay concise -- a position and a rate is a well. A
        `dict` of records names them by its keys (as `name` does otherwise), and
        the constructor takes the same thing:

        >>> from TPFA_ResSim import ResSim
        >>> model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, wells={
        ...     "I1": dict(xy=[0, 0], rate=+1),
        ...     "P1": dict(xy=[1, 1], rate=-1, rw=1e-3),
        ... })
        >>> model.wells.names
        ['I1', 'P1']
        >>> model.wells.rates
        array([[ 1.],
               [-1.]])
        >>> model.wells.WI.round(3)  # `P1` alone asked for a well model
        array([  nan, 2.498])

        A `path` becomes several completions of a single well, whose rate it
        shares out (ref `well_path`) and whose name they share:

        >>> model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10, wells=[
        ...     dict(name="I1", path=[[.05, .05], [.45, .05]], rate=+1, rw=1e-2),
        ...     dict(name="P1", xy=[.95, .95], rate=-1),
        ... ])
        >>> model.wells.nComp, model.wells.nWell
        (6, 2)
        >>> model.wells.group
        array([0, 0, 0, 0, 0, 1])
        >>> model.wells.rates.ravel().round(3)
        array([ 0.125,  0.25 ,  0.25 ,  0.25 ,  0.125, -1.   ])

        Schedules and control modes may be mixed freely across the wells: a
        constant is broadcast to the length of the longest schedule, while a
        spec that no well varies in time stays a singleton, which
        `TPFA_ResSim.ResSim.well_controls` reads at any `k`. A BHP-controlled
        well leaves the (ignored) `0` in `Wells.rates`, and `nan` marks the
        rate-controlled ones in `Wells.bhp` -- the conventions that the specs,
        being shared arrays, call for (ref `Wells.rates`).

        .. note:: A well must be given a control (`rate` and/or `bhp`).

            `rate=0` shuts it in. This is deliberate: an uncontrolled well would
            silently be a shut one.

        .. note:: The records are *not* retained.

            The arrays they produce are the whole of the configuration, so
            there is nothing to fall out of step with a subsequent edit of them
            (which is what the `repr` therefore reports).
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
                    f"Well '{name}': give it `xy` or `path`, not both."
                )
                assert rw is not None, f"Well '{name}': a `path` requires `rw`."
                _xy, _WI, _ = well_path(model, path, rw, skin)
            else:
                assert "xy" in spec, f"Well '{name}': give it an `xy` (or a `path`)."
                _xy = np.array(spec.pop("xy"), float).reshape((-1, 2))
                _WI = (
                    np.full(len(_xy), np.nan)
                    if rw is None
                    else peaceman_WI(model, _xy, rw, skin)
                )
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
            return cls()

        def stack(specs):
            """Stack the wells' `(nComp_i, nTime_i)` specs, widening the constants.

            NB: `at_time` broadcasts a *wholly* singleton spec, but the array is
            shared, so a well held constant beside a scheduled one is widened here.
            """
            nTime = max(spec.shape[1] for spec in specs)
            assert all(spec.shape[1] in [1, nTime] for spec in specs), (
                "The wells' schedules must be of equal length (or constant):"
                f" got {sorted({spec.shape[1] for spec in specs})}."
            )
            return np.vstack([np.broadcast_to(s, (len(s), nTime)) for s in specs])

        WI = np.concatenate(WI)
        return cls(
            # NB: `xy` first -- it is what defines `nComp`, by which the
            # `__setattr__` normalization shapes the others.
            xy     = np.vstack(xy),
            # Leave a spec unset (`None`) if no well made use of it
            rates  = stack(rates) if "rate" in specified else None,
            bhp    = stack(bhp) if "bhp" in specified else None,
            WI     = WI if np.isfinite(WI).any() else None,
            group  = np.concatenate(group),
            names  = names,
        )  # fmt: off
