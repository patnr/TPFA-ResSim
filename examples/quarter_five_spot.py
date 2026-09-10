"""Reproduce Fig. 6 of the reference paper, i.e. listing 9 -- and then vary it.

This runs the familiar 5-spot well pattern on a homogeneous and isotropic permeability
which, thanks to symmetry, only requires computing one of the 4 quadrants,
giving it the quarter-five spot problem.

There are some minor discrepancies compared with their Fig. 6.

- They claim to plot the initial pressure, but it rather seems like the final one to me.
- They panels portray `t` values are not available from the chosen time steps.
- Their water front has a corner that is more protruding (not due to the previous issue).

However, since we generate the very same output as the matlab code, we believe
the issue lies with the description in the paper, not with any error in the code.

The same flood is then run twice more: with the implicit transport scheme (whose
output is likewise verified against matlab), and with **scheduled** rates. In the
latter, a second injector (NW corner) shares the load: for the first 10 steps all of
the water enters from the SW corner, thereafter the two injectors split it equally,
and the front is correspondingly lopsided. A rate may be specified as a *schedule*
(an array over time), as done there -- whereupon the wells held constant get
broadcast along with it -- or (for feedback control, e.g. shutting wells on water
breakthrough) by overriding `ResSim.well_controls`.

In the figures:

- "Fig. 6": the water, injected in the SW corner, advances on the producer in
  the NE, in the shape that gives the quarter five-spot its name -- essentially
  that of the isobars of the first panel.
- "final saturation": the implicit scheme is the more diffusive of the two: its
  transition band ($0.2 < S < 0.8$) is some 15% wider than the explicit one's.
  Under the schedule, the front is no longer symmetric about the diagonal: the
  NW injector, switched on late, has flooded a band along the north edge, pinching
  the SW injector's front, which fills the rest. The quadrant *between* the two
  injectors is thus swept hardest (mean saturation 0.80 in the NW, against 0.57 in
  the SE; 0.66 in each for the base case).
- "schedule": the schedule itself (left), and the oil saturation in the producer
  (right). Both floods break through only in the last two of the 28 steps, the
  scheduled one a step ahead: its water arrives along the north edge.
"""

from mpl_tools.place import freshfig
import numpy as np

from minires import ResSim
from minires.plotting import show

## Setup
grid: dict = dict(Lx=1, Ly=1, Nx=64, Ny=64)
# Fluid properties are left at their defaults: vw = vo = 1, swc = sor = 0.

# The base case: an injector (SW) and a producer (NE), at constant, opposite rates.
# A well is a record of its position and its (signed) rate; ref `ResSim.wells`.
model = ResSim(**grid, wells=[dict(name="SW", xy=[0, 0], rate=+1),
                              dict(name="NE", xy=[1, 1], rate=-1)])

water_sat0 = model.fluid.swc * np.ones(model.Nxy)
nSteps = 28
dt = 0.7/nSteps

# Scheduled: the SW injector carries everything until step 10, then they share.
rate_sw = .5*np.ones(nSteps)
rate_nw = .5*np.ones(nSteps)
rate_sw[:10] = 1
rate_nw[:10] = 0
model_sch = ResSim(**grid, wells=[
    dict(name="SW", xy=[0, 0], rate=+rate_sw),
    dict(name="NW", xy=[0, 1], rate=+rate_nw),
    dict(name="NE", xy=[1, 1], rate=-1),   # constant: broadcast to the schedule
])

## Simulate
SS_exp, PP_exp = model.sim(dt, nSteps, water_sat0, pbar=False)
SS_imp, PP_imp = model.sim(dt, nSteps, water_sat0, implicit=True, pbar=False)
SS_sch, PP_sch = model_sch.sim(dt, nSteps, water_sat0, pbar=False)

## Plot: the paper's Fig. 6 (from the explicit scheme)
kws = dict(levels=17, cmap="jet", origin=None, extent=(0, model.Lx, 0, model.Ly))

fig, axs = freshfig("Fig. 6", nrows=2, ncols=3, sharex=True, sharey=True,
                    subplot_kw={'aspect': 'equal'})

for ax, t in zip(axs.ravel(), [None, .14, .28, .42, .56, .70]):
    if ax.get_subplotspec().is_last_row() : ax.set_xlabel("x")  # noqa
    if ax.get_subplotspec().is_first_col(): ax.set_ylabel("y")  # noqa

    if t is None:
        ax.set_title("Pressure")
        [P, V] = model.pressure_step(SS_exp[-1])  # Final+1 pressure
        ax.contourf(P.reshape(model.shape).T, **kws)

    else:
        k = int(t/dt)
        ax.set_title("t = {:.2f}".format(k * dt))
        Z = SS_exp[k].reshape(model.shape).T  # transpose/flip for plot orientation

        # Puts the values in gridcell centers (agrees w/ finite-vol. interpretation)
        # ax.imshow(Z[::-1], **kws)

        # Also colocates with gridcell centers, but does not extend to edges.
        # ax.contourf(Z, levels=17, cmap="jet", origin="lower")

        # Artificially stretches the field
        ax.contourf(Z, **kws)

fig.tight_layout()

## Plot: scheme and schedule comparison (at the final time)
fig, axs = freshfig("Quarter five-spot -- final saturation", ncols=3,
                    sharex=True, sharey=True, subplot_kw={'aspect': 'equal'})

for ax, (S, title) in zip(axs, [(SS_exp[-1], "Explicit (upwind)"),
                                (SS_imp[-1], "Implicit (Newton)"),
                                (SS_sch[-1], "Explicit, scheduled rates")]):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.contourf(S.reshape(model.shape).T, **kws)
axs[0].set_ylabel("y")
# The implicit scheme is (as usual) the more diffusive: its front is smeared.

fig.tight_layout()

## Plot: the schedule itself, and the resulting production
fig, axs = freshfig("Quarter five-spot -- schedule", ncols=2, figsize=(9, 3.5))

tt = dt*(1 + np.arange(nSteps))
for i, rate in enumerate(model_sch.wells.actual_rates[:2]):
    x, y = model_sch.wells.xy[i]
    name = model_sch.wells.names[i]
    axs[0].step(tt, rate, where="post", label=f"Injector {name} @ ({x:.2f}, {y:.2f})")
axs[0].set(title="Injection rates", xlabel="Time", ylabel="Rate", ylim=(-.05, 1.05))
axs[0].legend()

prd = model.xy2ind(*model.wells.xy[1])  # the NE producer's cell (in both models)
model.plt_production(axs[1], np.column_stack([SS_exp[1:, prd], SS_sch[1:, prd]]),
                     finalize=False, labels=["NE (base case)", "NE (scheduled)"])
fig.tight_layout()

## Animation
animation = model.anim(SS_exp, SS_exp[1:, [prd]])

# Regression values, checked by `tests/test_examples.py`.
# The sub-sampling `[::600]` matches that of the matlab reference values.
__digest__ = dict(explicit  = SS_exp[-1, ::600],
                  implicit  = SS_imp[-1, ::600],
                  scheduled = SS_sch[-1, ::600])

if __name__ == "__main__":
    show()
