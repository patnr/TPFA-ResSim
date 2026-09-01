"""Reproduce Fig. 1 of the reference paper, i.e. example 1.

Solves the (single, elliptic) pressure equation on a uniform permeability field
(coarse grid) and on a random, smoothed, log-normal one (finer grid).

There are some discrepancies, of course, because of

- The use of random numbers
- Differences in scipy's `smooth` and matlab
- It appears they've translated the pressure field to be positive
  (in their panels it seems to have minimum value 0).
  As a *velocity* potential, this should not matter.
  Indeed, since `ct == 0` here, the pressure is only defined up to a constant;
  ref `examples/pressure_diffusion.py`.

In the figure:

- Left panel: with uniform permeability, the isobars are smooth arcs,
  centred on the wells in the SW and NE corners.
- Right panel: the same solve, but on the (smoothed, log-normal) permeability
  of the middle panel. The isobars are now distorted, bunching up over the
  low-permeability regions -- the correlation of $\\log |∇p|$ with $\\log K$ is
  about $-0.4$ -- since the tighter the rock, the steeper the gradient needed
  to pass a given flux.
"""

from mpl_tools.place import freshfig
import numpy as np
import numpy.random as rnd
from scipy.ndimage import uniform_filter as smooth

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

rnd.seed(4)  # Reproducibility (the values are regression tested)

fig, axs = freshfig("Fig. 1", ncols=3, nrows=2, gridspec_kw={'height_ratios': (9, 1)})

## Panel 0: uniform permeability, coarse grid
model = ResSim(Lx=1, Ly=1, Nx=8, Ny=8,
               well_xy=[[0, 0], [1, 1]], well_rates=[[1], [-1]])

model.assemble_wells(None, None, 0)
[P_coarse, V] = model.TPFA(model.K)

ax = axs[0, 0]
ax.set(title="Pressure", aspect="equal")
cc = ax.contourf(P_coarse.reshape(model.shape).T, levels=17, cmap="jet")
ax.contour(P_coarse.reshape(model.shape).T, levels=17)
cb = fig.colorbar(cc, axs[1, 0], orientation="horizontal")
cb.ax.tick_params(labelsize=8)

## Panels 1 and 2: heterogeneous permeability, finer grid
model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32,
               well_xy=[[0, 0], [1, 1]], well_rates=[[1], [-1]])
logK = 5*smooth(smooth(rnd.randn(2, *model.shape)))
model.K = np.exp(logK)

ax = axs[0, 1]
ax.set(title="log-Permeability", aspect="equal")
# ax.imshow(K.T[::-1, :, 0], cmap="jet")
cc = ax.pcolormesh(logK.T[..., 0], edgecolors='k', linewidth=.01, cmap="jet")
fig.colorbar(cc, axs[1, 1], orientation="horizontal")

model.assemble_wells(None, None, 0)
[P_fine, V] = model.TPFA(model.K)

ax = axs[0, 2]
ax.set(title="Pressure", aspect="equal")
cc = ax.contourf(P_fine.reshape(model.shape).T, levels=17, cmap="jet")
ax.contour(P_fine.reshape(model.shape).T, levels=17)
cb = fig.colorbar(cc, axs[1, 2], orientation="horizontal")
cb.ax.tick_params(labelsize=8)

fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(pres_coarse = P_coarse,
                  pres_fine   = P_fine)

if __name__ == "__main__":
    show()
