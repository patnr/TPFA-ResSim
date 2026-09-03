"""Tools for working with model grid coordinates.

Most functions here are barely in use, mostly serving as reference.
After all, it is surprisingly hard to remember
which direction and index is for x and which is for y.

The index ordering is "C-style" (numpy default).
This choice means that `x` is the 1st coord., `y` is 2nd,
and is hardcoded in the reservoir simulator model code
(in what takes place **between** `np.ravel` and `np.reshape`,
both of which are configured to use row-major index ordering.
"F-style" (column-major) indexing implementation is perfectly possible,
but would imply an undue amount hassle).
Conveniently, it also means that `x` and `y` tend to occur in alphabetic order.
Thus, in printing a matrix of a field, the `x` coordinate corresponds to the row index.
By contrast, the plotting module depicts `x` from left to right, `y` from bottom to top.
"""

from dataclasses import dataclass
from typing import overload

import numpy as np
import numpy.typing as npt


@dataclass
class Grid2D:
    """Defines a 2D rectangular grid.

    Example (2 x-nodes, 5 y-nodes):
    >>> grid = Grid2D(Lx=6, Ly=10, Nx=3, Ny=5)

    The nodes are centered in the cells:
    >>> X, Y = grid.mesh
    >>> X
    array([[1., 1., 1., 1., 1.],
           [3., 3., 3., 3., 3.],
           [5., 5., 5., 5., 5.]])

    You can compute cell boundaries (i.e. non-central nodes) by adding or subtracting
    `hx`/2 and `hy`/2 (i.e. you will miss either boundary at 0 or `Lx` or `Ly`).

    Test of round-trip capability of grid mapping computations:
    >>> ij = (0, 4)
    >>> grid.xy2sub(X[ij], Y[ij]) == ij
    array([ True,  True])

    >>> grid.sub2xy(*ij) == (X[ij], Y[ij])
    array([ True,  True])
    """

    Lx: float = 1.0
    """Physical x-length of domain."""
    Ly: float = 1.0
    """Physical y-length of domain."""
    Nx: int = 32
    """Number of grid cells (and their centres) in x dir."""
    Ny: int = 32
    """Number of grid cells (and their centres) in y dir."""

    @property
    def shape(self) -> tuple:
        """`(Nx, Ny)`"""
        return self.Nx, self.Ny

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(np.prod(self.shape))

    @property
    def domain(self) -> tuple:
        """`((0, 0), (Lx, Ly))`"""
        return ((0, 0), (self.Lx, self.Ly))

    @property
    def Nxy(self) -> int:
        """`Nx` * `Ny`"""
        return int(np.prod(self.shape))

    @property
    def hx(self) -> float:
        """x-length of cells"""
        return self.Lx / self.Nx

    @property
    def hy(self) -> float:
        """y-length of cells"""
        return self.Ly / self.Ny

    @property
    def h2(self) -> float:
        """`hx` * `hy`"""
        return self.hx * self.hy

    @property
    def mesh(self) -> tuple:
        """Generate 2D coordinate grid of cell centres."""
        xx = np.linspace(0, self.Lx, self.Nx, endpoint=False) + self.hx / 2
        yy = np.linspace(0, self.Ly, self.Ny, endpoint=False) + self.hy / 2
        return np.meshgrid(xx, yy, indexing="ij")

    def sub2ind(
        self, ix: int | np.ndarray, iy: int | np.ndarray
    ) -> np.intp | np.ndarray:
        """Convert index `(ix, iy)` to index in flattened array."""
        idx = np.ravel_multi_index((ix, iy), self.shape)
        return idx

    def ind2sub(self, ind: int | np.intp | np.ndarray) -> np.ndarray:
        """Inv. of `self.sub2ind`."""
        ix, iy = np.unravel_index(ind, self.shape)
        return np.asarray([ix, iy])

    def xy2sub(self, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray:
        """Convert physical coordinate tuple to `(ix, iy)`, ix ∈ {0, ..., Nx-1}.

        .. warning:: `xy2sub` and `xy2ind` *round* to nearest cell center.
            I.e. they are not injective.
            The alternative would be to return some kind
            of interpolation weights distributing `(x, y)` over multiple nodes.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        # Don't silence errors! Validation is useful in optimisation (e.g.)
        assert np.all(x <= self.Lx)
        assert np.all(y <= self.Ly)
        # Set upper border values to slightly interior.
        # NB: the nudge is *relative*, `Lx` being of whatever magnitude the
        # units imply (ref `TPFA_ResSim.ResSim.cdarcy`).
        x = x.clip(max=self.Lx * (1 - 1e-12))
        y = y.clip(max=self.Ly * (1 - 1e-12))
        ix = np.floor(x / self.Lx * self.Nx).astype(int)
        iy = np.floor(y / self.Ly * self.Ny).astype(int)
        return np.asarray([ix, iy])

    # Overloaded so that the (array-valued) well lookups of `ResSim` type-check
    @overload
    def xy2ind(self, x: float, y: float) -> np.intp: ...
    @overload
    def xy2ind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def xy2ind(self, x: npt.ArrayLike, y: npt.ArrayLike) -> np.intp | np.ndarray:
        """Convert physical coord to flat indx."""
        i, j = self.xy2sub(x, y)
        return self.sub2ind(i, j)

    def sub2xy(self, ix: npt.ArrayLike, iy: npt.ArrayLike) -> np.ndarray:
        """Inverse of `self.xy2sub`."""
        x = (np.asarray(ix) + 0.5) * self.hx
        y = (np.asarray(iy) + 0.5) * self.hy
        return np.asarray([x, y])

    def ind2xy(self, ind: int | np.intp | np.ndarray) -> np.ndarray:
        """Inverse of `self.xy2ind`."""
        i, j = self.ind2sub(ind)
        return self.sub2xy(i, j)

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
