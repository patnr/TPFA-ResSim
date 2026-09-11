"""The two-phase fluid, held at `minires.ResSim.fluid`.

`Fluid` holds the phase viscosities and the parameters of the Corey relative
permeability curves, and computes what the model needs of them: the mobilities
(`RelPerm`, Listing 6 of the reference paper) and the fractional flow
(`fractional_flow`), each with its derivative (`dRelPerm`, `dfractional_flow`).
Curves of another shape (e.g. tabulated) are a subclass overriding `RelPerm` and
`dRelPerm`.

## Theory

Fossil fuel hydrocarbons is sedimented, pressurized, organic material
(mostly plants?) that used to live on the **sub-sea** continental shelves.
**On-land** organic material turns into coal.
⇒ Saudi-Arabia used to be sub-sea?
The *energy* in oil & gas comes from the sun (photosynthesis),
not the compression.
The lightest *hydrocarbons* (methane, ethane, etc.) usually escapes quickly,
while oils moves slowly towards the surface.
Sometimes the geology is bends to form caps of non-permeable rock
(ref `minires.ResSim.K`), so that the migrating hydrocarbons are trapped.
In the *North Sea*, these reservoirs lie 1000-3000 meters below the sea bed.
Norway is also surrounded by the *Norwegian sea*,
and the *Barents sea*, towards Murmansk.

Reservoir simulators implement porous media flow
on upscaled geophysical parameters typically with grid blocks between 1 - 100 m.
They usually parameterize multiphase flow.
If only the two phases of oil and water are used it is called **black-oil**.
A common assumption is that the flow is **immiscible**: not mixing (oil and water).
But this does not mean that gas cannot be *dissolved* in oil.

The **phases** (water, oil, gas), whose saturations sum to $1$,
contains *components* (e.g. methane, ethane, propane),
usually grouped as pseudo-components.
Each phase's *mass fraction* component, $c_{phase,i}$, sums to $1$.
Each phase has **density**, $ρ$ and **viscosity**, $μ$ (`Fluid.vw`, `Fluid.vo`),
generally functions of the phase **pressure**,
but usually neglected except for gas.
The differences in pressure are named **capillary pressure**
because they arise due to **interfacial tensions**; this model has none.
A phase's **compressibility** is defined similar as for the rock's
(ref `minires.ResSim.ct`, which lumps them all into one).
Confusingly, it is also denoted with $c$, but using only a single subscript.

Phases do not really mix. But in macro-scale modelling all phases
may be present at the same location. Therefore a phase's permeability
should depend on the saturations, to which end we introduce *relative permeability*,
$k_{r,i} = k_{r,i}(s_g, s_o), i = g, o, w$
a nonlinear function, yielding an (effective) permeability
$\\mathbf{K_i} = \\mathbf{K} k_{r,i}$
Relative permeability curves do not extend all over the interval $[0, 1]$.
The smallest saturation where a phase is mobile is called the
**residual saturation** (`Fluid.swc`, `Fluid.sor`).
This *adsorption* effects may vary, and this may have important effects,
particularly for simulation of *polymer injection*.
The uncertainty regarding relative permeability is modest compared to
the enormous uncertainty of the rock permeability (`minires.ResSim.K`).

Everything depends on *thermodynamics*, but this is often complex and neglected,
except perhaps for the bubble/boiling point pressures,
which govern how much of the gas dissolves in oil.

Since *compressibility* relates volumes to pressure,
a volume must be qualified by where it is measured.
The **formation volume factor**, $B$, is the ratio of the volume at reservoir
conditions to that of the same mass at the surface ("stock tank"),
and is how field rates (measured at the surface) are converted
to the reservoir rates that a simulator works in. This model has $B = 1$.
Related **PVT** (pressure-volume-temperature) vocabulary:
the **bubble point** is the pressure below which gas comes out of solution;
an oil above it is **undersaturated**, and the amount of gas it holds is the
*solution gas-oil ratio*, $R_s$.
"""

from dataclasses import dataclass

import numpy as np

from minires._repr import AlignedRepr


@dataclass
class Fluid(AlignedRepr):
    """A two-phase (water/oil) fluid: viscosities and Corey relative permeabilities.

    The relative permeabilities are Corey (power-law) curves of the normalized
    saturation $ S $ (`rescale_sat`),

    $$ k_{rw} = k_{rw}^0 \\, S^{n_w} \\,, \\qquad k_{ro} = k_{ro}^0 \\, (1 - S)^{n_o} \\,,
       \\qquad S = \\frac{s - S_{wc}}{1 - S_{wc} - S_{or}} \\,, $$

    with $ S $ clipped to $[0, 1]$, so that a phase below its residual saturation
    is immobile (rather than mobile with the wrong sign, as an odd power would
    make it). The defaults give the quadratic curves of the reference paper
    (Listing 6), $ S^2 $ and $ (1 - S)^2 $, and unit viscosities.

    >>> fluid = Fluid(vo=2, swc=.2, sor=.2)
    >>> Mw, Mo = fluid.RelPerm(np.array([.1, .2, .5, .8, .9]))
    >>> Mw.round(4), Mo.round(4)
    (array([0.  , 0.  , 0.25, 1.  , 1.  ]), array([0.5  , 0.5  , 0.125, 0.   , 0.   ]))
    """

    __repr__ = AlignedRepr.__repr__

    vw: float = 1.0
    """Viscosity for water."""
    vo: float = 1.0
    """Viscosity for oil."""
    swc: float = 0.0
    """Irreducible saturation, water: $ k_{rw} = 0 $ below it."""
    sor: float = 0.0
    """Irreducible saturation, oil: $ k_{ro} = 0 $ above $ s = 1 - S_{or} $."""
    nw: float = 2.0
    """Corey exponent of the water relative permeability."""
    no: float = 2.0
    """Corey exponent of the oil relative permeability."""
    krw0: float = 1.0
    """End-point (maximal) water relative permeability, at $ s = 1 - S_{or} $."""
    kro0: float = 1.0
    """End-point (maximal) oil relative permeability, at $ s = S_{wc} $."""

    def rescale_sat(self, s: np.ndarray) -> np.ndarray:
        """The normalized saturation $ S $ (unclipped). Ref paper, p. 32."""
        return (s - self.swc) / (1 - self.swc - self.sor)

    # RelPerm() -- listing 6
    def RelPerm(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rel. permeabilities of water and oil, as mobilities (perm/viscosity)."""
        S = np.clip(self.rescale_sat(s), 0, 1)
        Mw = self.krw0 * S**self.nw / self.vw
        Mo = self.kro0 * (1 - S) ** self.no / self.vo
        return Mw, Mo

    def dRelPerm(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Derivatives of `RelPerm` wrt `s` (zero where the curves are clipped)."""
        S = self.rescale_sat(s)
        inside = (0 <= S) & (S <= 1)  # one-sided at the ends, as the reference code
        S = np.clip(S, 0, 1)
        w = 1 - self.swc - self.sor
        dMw = np.where(inside, self.krw0 * self.nw * S ** (self.nw - 1) / w, 0) / self.vw
        dMo = -np.where(inside, self.kro0 * self.no * (1 - S) ** (self.no - 1) / w, 0) / self.vo
        return dMw, dMo

    def fractional_flow(self, s: np.ndarray) -> np.ndarray:
        """The water fractional flow, $ f_w = λ_w / λ_t $."""
        Mw, Mo = self.RelPerm(s)
        return Mw / (Mw + Mo)

    def dfractional_flow(self, s: np.ndarray) -> np.ndarray:
        """Derivative of `fractional_flow` wrt `s`.

        Separate from `fractional_flow` (rather than a second output of it) because
        the explicit transport scheme wants $ f_w $ alone, in its innermost loop,
        where the derivative would cost as much again as everything else.
        """
        Mw, Mo = self.RelPerm(s)
        dMw, dMo = self.dRelPerm(s)
        return (dMw * Mo - Mw * dMo) / (Mw + Mo) ** 2
