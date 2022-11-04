# Reservoir simulator

A 2D, two-phase, black-oil, immiscible, incompressible
reservoir simulator
using TPFA (two-point flux approximation)
and explicit time stepping.

Based on [Matlab codes](http://folk.ntnu.no/andreas/papers/ResSimMatlab.pdf)
from NTNU/Sintef by Jørg E. Aarnes, Tore Gimse, and Knut–Andreas Lie.  
Translated to Python by Patrick N. Raanes.

The Python code produces the same output as the Matlab version
(up to errors from the linear solvers).
Main changes:

- `83293bcb`: Converted from 3D to 2D for simplicity.
- `27208d5d`: Index ordering is C-major (numpy standard), not F-major.
- `7543f574`: Vectors are "numpy-thonic", in using 1d arrays, not (2d) columns.
- `cade3156`: Several linear solvers suggested.
- `f33c571a`: OOP
- `55ce7325`: Facilities for working on the grid.
- `e0d12b06`: Convenient well arranger (ensures total sink + source = 0).

## Used in

- [History matching tutorial](https://github.com/patnr/HistoryMatching)
