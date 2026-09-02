"""Reproduce Fig. 6 of the reference paper, i.e. listing 9.

This runs the familiar 5-spot well pattern on a homogeneous and isotropic permeability
which, thanks to symmetry, only requires computing one of the 4 quadrants,
giving it the quarter-five spot problem.

There are some minor discrepancies compared with their Fig. 6.

- They claim to plot the initial pressure, but it rather seems like the final one to me.
- They panels portray `t` values are not available from the chosen time steps.
- Their water front has a corner that is more protruding (not due to the previous issue).

However, since we generate the very same output as the matlab code, we believe
the issue lies with the description in the paper, not with any error in the code.

The last figure compares the two transport schemes (whose outputs are both
verified against matlab) as well as a doubled-rate run.

In the figures:

- "Fig. 6": the water, injected in the SW corner, advances on the producer in
  the NE, in the shape that gives the quarter five-spot its name -- essentially
  that of the isobars of the first panel.
- "final saturation": the implicit scheme is the more diffusive of the two: its
  transition band ($0.2 < S < 0.8$) is some 15% wider than the explicit one's.
  Doubling the rates, meanwhile, simply gets further in the same time
  (mean saturation 0.86, against 0.70).
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup
grid = dict(Lx=1, Ly=1, Nx=64, Ny=64)
# Fluid properties are left at their defaults: vw = vo = 1, swc = sor = 0.


def wells(q):
    """A well is a record of its position and its (signed) rate.

    Ref `ResSim.wells`. Here: an injector (SW) and a producer (NE).
    """
    return [dict(xy=[0, 0], rate=+q),
            dict(xy=[1, 1], rate=-q)]



model = ResSim(**grid, wells=wells(1))
model_2x = ResSim(**grid, wells=wells(2))

water_sat0 = model.swc * np.ones(model.Nxy)
nSteps = 28
dt = 0.7/nSteps

## Simulate
SS_exp, PP_exp = model.sim(dt, nSteps, water_sat0, pbar=False)
SS_imp, PP_imp = model.sim(dt, nSteps, water_sat0, implicit=True, pbar=False)
SS_2x , PP_2x  = model_2x.sim(dt, nSteps, water_sat0, pbar=False)

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

## Plot: scheme and rate comparison (at the final time)
fig, axs = freshfig("Quarter five-spot -- final saturation", ncols=3,
                    sharex=True, sharey=True, subplot_kw={'aspect': 'equal'})

for ax, (S, title) in zip(axs, [(SS_exp[-1], "Explicit (upwind)"),
                                (SS_imp[-1], "Implicit (Newton)"),
                                (SS_2x[-1] , "Explicit, 2x rates")]):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.contourf(S.reshape(model.shape).T, **kws)
axs[0].set_ylabel("y")
# The implicit scheme is (as usual) the more diffusive: its front is smeared.

fig.tight_layout()

## Animation
prod = [model.xy2ind(*model.wells.xy[1])]
animation = model.anim(SS_exp, SS_exp[1:, prod])

# Regression values, checked by `tests/test_examples.py`.
# The sub-sampling `[::600]` matches that of the matlab reference values.
__digest__ = dict(explicit = SS_exp[-1, ::600],
                  implicit = SS_imp[-1, ::600],
                  doubled  = SS_2x[-1, ::600])

if __name__ == "__main__":
    show()
