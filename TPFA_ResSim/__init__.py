""".. include:: README.md"""

from TPFA_ResSim.core import ResSim
from TPFA_ResSim.grid import Fluxes, Grid2D
from TPFA_ResSim.fluids import Fluid
from TPFA_ResSim.wells import Wells, aquifer_WI, peaceman_WI, well_path

# Also pdoc's table of contents: `ResSim` is documented on the package page (beside the
# README), and the listed submodules on their own pages. `core` is deliberately absent,
# lest `ResSim` be documented twice; so are the other re-exports, which are documented
# in their home modules.
__all__ = ["ResSim", "grid", "wells", "fluids", "plotting", "tlm"]
