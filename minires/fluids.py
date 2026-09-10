"""The two-phase fluid, held at `minires.ResSim.fluid`.

`Fluid` holds the phase viscosities and the parameters of the Corey relative
permeability curves, and computes what the model needs of them: the mobilities
(`RelPerm`, Listing 6 of the reference paper) and the fractional flow
(`fractional_flow`), each with its derivative (`dRelPerm`, `dfractional_flow`).
Curves of another shape (e.g. tabulated) are a subclass overriding `RelPerm` and
`dRelPerm`.
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
