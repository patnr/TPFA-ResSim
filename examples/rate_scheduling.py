"""Steering the water front with time-varying injection rates.

Two injectors (SW and NW corners) feed a single producer (NE corner).
Their rates are *scheduled*: for the first 10 steps all of the water enters from
the SW corner, thereafter the two injectors share the load equally.
The resulting front is correspondingly lopsided.

A rate may be specified as a *schedule* (an array over time), as done here --
whereupon the wells held constant get broadcast along with it -- or (for
feedback control, e.g. shutting wells on water breakthrough) by overriding
`ResSim.well_controls`.

In the figures:

- "saturation": at t = 0.125, and still at t = 0.25 (the moment of the switch),
  all of the water has come from the SW. By t = 0.45 a second front has grown
  from the NW injector, and by t = 0.70 the two have merged into a distinctly
  lopsided sweep -- compare the symmetric front of `examples.quarter_five_spot`.
- "wells": the schedule itself (left), and the oil saturation in the producer
  (right), which shows that the water only breaks through at the very end
  (step 27 of 28).
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup
nSteps = 28
dt = 0.7/nSteps

# Schedule: the SW injector carries everything until step 10, then they share.
rate_sw = .5*np.ones(nSteps)
rate_nw = .5*np.ones(nSteps)
rate_sw[:10] = 1
rate_nw[:10] = 0

model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64, wells=[
    dict(name="SW", xy=[0, 0], rate=+rate_sw),
    dict(name="NW", xy=[0, 1], rate=+rate_nw),
    dict(name="NE", xy=[1, 1], rate=-1),   # constant: broadcast to the schedule
])

water_sat0 = model.swc * np.ones(model.Nxy)

## Simulate
SS, PP = model.sim(dt, nSteps, water_sat0, pbar=False)

## Plot: saturation snapshots
fig, axs = freshfig("Rate scheduling -- saturation", nrows=2, ncols=2,
                    sharex=True, sharey=True)
for ax, k in zip(axs.ravel(), [5, 10, 18, nSteps]):
    model.plt_field(ax, SS[k], "oil", finalize=False, colorbar=False,
                    title=f"t = {k*dt:.2f}")
fig.tight_layout()

## Plot: the schedule itself, and the resulting production
fig, axs = freshfig("Rate scheduling -- wells", ncols=2, figsize=(9, 3.5))

tt = dt*(1 + np.arange(nSteps))
for i, rate in enumerate(model.wells.actual_rates[:2]):
    x, y = model.wells.xy[i]
    name = model.wells.names[i]
    axs[0].step(tt, rate, where="post", label=f"Injector {name} @ ({x:.2f}, {y:.2f})")
axs[0].set(title="Injection rates", xlabel="Time", ylabel="Rate", ylim=(-.05, 1.05))
axs[0].legend()

prd = [model.xy2ind(*model.wells.xy[2])]
model.plt_production(axs[1], SS[1:, prd], finalize=False,
                     labels=model.wells.names[2:])
fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(sat_final=SS[-1, ::600])

if __name__ == "__main__":
    show()
