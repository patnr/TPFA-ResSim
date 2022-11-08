"""Reproduce Fig. 1 of the reference paper, i.e. example 1.

There are some discrepancies, of course, because of

- The use of random numbers
- The workings of `smooth`
- It appears they've translated the pressure field to be positive
  (in their panels it seems to have minimum value 0).
  As a *velocity* potential, this should not matter.
"""

from mpl_tools.place import freshfig
import numpy as np
import numpy.random as rnd
from scipy.ndimage import uniform_filter as smooth
import matplotlib.pyplot as plt

from TPFA_ResSim import ResSim, recurse

plt.ion()
# rnd.seed(4)
fig, axs = freshfig("Fig. 1", ncols=3, nrows=2, gridspec_kw={'height_ratios': (9, 1)})

## Panel 0
model = ResSim(Lx=1, Ly=1, Nx=8, Ny=8)
model.config_wells(inj =[[0, 0, 1]],
                   prod=[[1, 1, -1]])

[P, V] = model.TPFA(model.Gridded.K)

ax = axs[0, 0]
ax.set(title="Pressure", aspect="equal")
cc = ax.contourf(P.reshape(model.shape).T, levels=17, cmap="jet")
ax.contour(P.reshape(model.shape).T, levels=17)
cb = fig.colorbar(cc, axs[1, 0], orientation="horizontal")
cb.ax.tick_params(labelsize=8)

## Panels 1 and 2
model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32)
logK = 5*smooth(smooth(rnd.randn(2, *model.shape)))
model.Gridded.K = np.exp(logK)
model.config_wells(inj =[[0, 0, 1]],
                   prod=[[1, 1, -1]])

ax = axs[0, 1]
ax.set(title="Porosity", aspect="equal")
# ax.imshow(K.T[::-1, :, 0], cmap="jet")
cc = ax.pcolormesh(logK.T[..., 0], edgecolors='k', linewidth=.01, cmap="jet")
fig.colorbar(cc, axs[1, 1], orientation="horizontal")

[P, V] = model.TPFA(model.Gridded.K)

ax = axs[0, 2]
ax.set(title="Pressure", aspect="equal")
cc = ax.contourf(P.reshape(model.shape).T, levels=17, cmap="jet")
ax.contour(P.reshape(model.shape).T, levels=17)
cb = fig.colorbar(cc, axs[1, 2], orientation="horizontal")
cb.ax.tick_params(labelsize=8)

fig.tight_layout()
