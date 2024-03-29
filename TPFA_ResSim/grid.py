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

import numpy as np


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
    def shape(self):
        """`(Nx, Ny)`"""
        return self.Nx, self.Ny

    @property
    def size(self):
        """Total number of elements."""
        return np.prod(self.shape)

    @property
    def domain(self):
        """`((0, 0), (Lx, Ly))`"""
        return ((0, 0), (self.Lx, self.Ly))

    @property
    def Nxy(self):
        """`Nx` * `Ny`"""
        return np.prod(self.shape)

    @property
    def hx(self):
        """x-length of cells"""
        return self.Lx/self.Nx

    @property
    def hy(self):
        """y-length of cells"""
        return self.Ly/self.Ny

    @property
    def h2(self):
        """`hx` * `hy`"""
        return self.hx*self.hy

    @property
    def mesh(self):
        """Generate 2D coordinate grid of cell centres."""
        xx = np.linspace(0, self.Lx, self.Nx, endpoint=False) + self.hx/2
        yy = np.linspace(0, self.Ly, self.Ny, endpoint=False) + self.hy/2
        return np.meshgrid(xx, yy, indexing="ij")

    def sub2ind(self, ix, iy):
        """Convert index `(ix, iy)` to index in flattened array."""
        idx = np.ravel_multi_index((ix, iy), self.shape)
        return idx

    def ind2sub(self, ind):
        """Inv. of `self.sub2ind`."""
        ix, iy = np.unravel_index(ind, self.shape)
        return np.asarray([ix, iy])

    def xy2sub(self, x, y):
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
        # Set upper border values to slightly interior
        x = x.clip(max=self.Lx-1e-8)
        y = y.clip(max=self.Ly-1e-8)
        ix = np.floor(x / self.Lx * self.Nx).astype(int)
        iy = np.floor(y / self.Ly * self.Ny).astype(int)
        return np.asarray([ix, iy])

    def xy2ind(self, x, y):
        """Convert physical coord to flat indx."""
        i, j = self.xy2sub(x, y)
        return self.sub2ind(i, j)

    def sub2xy(self, ix, iy):
        """Inverse of `self.xy2sub`."""
        x = (np.asarray(ix) + .5) * self.hx
        y = (np.asarray(iy) + .5) * self.hy
        return np.asarray([x, y])

    def ind2xy(self, ind):
        """Inverse of `self.xy2ind`."""
        i, j = self.ind2sub(ind)
        return self.sub2xy(i, j)
