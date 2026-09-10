"""An irregular reservoir on the rectangular grid: an outline, and a sealing fault.

The grid stays rectangular; the reservoir need not. `active`
(ref `minires.ResSim.active`) marks the cells that take part; the rest are
inert -- closed off (zero flux across their faces), without equations, and
their state carried through unchanged -- as if they were not there.

- **An outline**: here an ellipse, cut out of a 48² grid. The flow goes around
  the (now curved) boundary as it would around the box.
- **A sealing fault**: a line of inactive cells. It need only be unbroken --
  a diagonal staircase suffices, fluxes passing through faces alone -- to block
  the flow entirely, which must then go around its tip: the front reaches the
  producer (top right) by the long way round, beneath the fault, and the region
  behind the fault is swept last.
  Should it cut the reservoir in two, each half is a reservoir of its own, and
  (being incompressible) must balance its own rates -- ref `active`.

In the figure: the permeability, painted cell by cell (`cellwise=True`), which
shows the mask exactly as it is; then the oil saturation at two times, as
contours -- these interpolate between cell centres, so they also leave blank
the half-cell margin around the inactive cells (as around the domain).
"""

from mpl_tools.place import freshfig
import numpy as np
from scipy.ndimage import uniform_filter as smooth

from minires import ResSim
from minires.plotting import show

rng = np.random.default_rng(3)  # Reproducibility (the values are regression tested)

## An irregular reservoir: an ellipse, with a sealing fault
model = ResSim(Lx=1, Ly=1, Nx=48, Ny=48,
               wells=[dict(xy=[.2, .5], rate=+1, name="Inj"),
                      dict(xy=[.8, .65], rate=-1, name="Prd")])
X, Y = model.mesh
outline = ((X - .5) / .47)**2 + ((Y - .5) / .4)**2 <= 1
# The fault: one cell per row (slope < 1 ⇒ an unbroken staircase), from y = .3 up
x_fault = .55 + .25 * (Y - .5)
fault = (abs(X - x_fault) <= model.hx / 2 + 1e-9) & (Y > .3)
model.active = outline & ~fault
logK = 4 * smooth(smooth(rng.standard_normal(model.shape)))
model.K = np.exp(logK)

dt = .01
nSteps = 60
S0 = np.zeros(model.Nxy)
SS, PP = model.sim(dt, nSteps, S0, pbar=False)

# The inactive cells are inert: their state is simply carried through.
inactive = ~model.active.ravel()
assert (SS[:, inactive] == 0).all() and (PP[:, inactive] == 0).all()
# The active ones conserve the water: what is injected and not yet produced is in place.
iprd = model.xy2ind(*model.wells.xy[1])
water_cut = model.fluid.fractional_flow(SS[:, iprd])
kBT = int(np.argmax(water_cut > 0))  # breakthrough: during step kBT-1 → kBT
assert 0 < kBT < nSteps
pv = model.pore_volume()[~inactive]
assert np.isclose((pv * SS[kBT - 1][~inactive]).sum(), (kBT - 1) * dt)

## Plot
fig, axs = freshfig("Inactive cells", ncols=3, figsize=(11, 3.8),
                    sharex=True, sharey=True)
kws: dict = dict(finalize=False, wells=dict(size=.5))
model.plt_field(axs[0], logK, cellwise=True, cmap="viridis", levels=17,
                title="log-Permeability (cells painted flat)", **kws)
for i, k in enumerate([nSteps // 3, nSteps]):
    model.plt_field(axs[1 + i], SS[k], "oil", colorbar=(i == 1), labels=False,
                    title=f"Oil saturation, t = {k*dt:.2f}", **kws)
fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(sat_final = SS[-1][~inactive],
                  p_final   = PP[-1][~inactive],
                  water_cut = water_cut)

if __name__ == "__main__":
    show()
