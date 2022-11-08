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
"""

from mpl_tools.place import freshfig
import numpy as np
import matplotlib.pyplot as plt
import pytest

from TPFA_ResSim import ResSim, recurse

model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64)
model.config_wells(inj =[[0, 0, 1]],
                   prod=[[1, 1, -1]])

# Change fluid properties (default: 1, 1, 0, 0)
# model.Fluid.vw  = 3e-4
# model.Fluid.vo  = 3e-3
# model.Fluid.swc = .2
# model.Fluid.sor = .2

water_sat0 = model.Fluid.swc * np.ones(model.M)
nSteps = 28
dt = 0.7/nSteps


## Tests
# Automatically run by pytest
@pytest.mark.parametrize("imp", [True, False])
def test_compare_final_output(imp):
    SS = recurse(model.time_stepper(dt, implicit=imp), nSteps, water_sat0)

    # From matlab_codes/listing9.m: >> S(1:600:end)
    matlab_output = ([0.99963, 0.91573, 0.75736, 0.89799, 0.79862, 0.62061, 0.62483]
                     if imp else
                     [0.99965, 0.91968, 0.77584, 0.90268, 0.81068, 0.67467, 0.66065])
    assert np.all(np.isclose(SS[-1, ::600], matlab_output))


if __name__ == "__main__":
    SS = recurse(model.time_stepper(dt, implicit=False), nSteps, water_sat0)

    ## Plot
    plt.ion()

    kws = dict(levels=17, cmap="jet", origin=None,
               extent=(0, model.Lx, 0, model.Ly))

    fig, axs = freshfig("Fig. 6", nrows=2, ncols=3, sharex=True, sharey=True,
                        subplot_kw={'aspect': 'equal'})

    for ax, t in zip(axs.ravel(), [None, .14, .28, .42, .56, .70]):
        if ax.get_subplotspec().is_last_row() : ax.set_xlabel("x")  # noqa
        if ax.get_subplotspec().is_first_col(): ax.set_ylabel("y")  # noqa

        if t is None:
            ax.set_title("Pressure")
            [P, V] = model.pressure_step(SS[-1])  # Final+1 pressure
            ax.contourf(P.reshape(model.shape).T, **kws)

        else:
            k = int(t/dt)
            ax.set_title("t = {:.2f}".format(k * dt))
            Z = SS[k].reshape(model.shape)  # Also transpose/flip for plot orientation

            # Puts the values in gridcell centers (agrees w/ finite-vol. interpretation)
            # ax.imshow(Z.T[::-1], **kws)

            # Also colocates with gridcell centers, but does not extend to edges.
            # ax.contourf(Z.T, levels=17, cmap="jet", origin="lower")

            # Artificially stretches the field
            ax.contourf(Z.T, **kws)


    ## Animation
    from TPFA_ResSim import plotting
    plotting.model = model
    plotting.coord_type = "absolute"
    prod = [model.xy2ind(x, y) for (x, y, _) in model.producers]
    animation = plotting.anim(None, SS, SS[1:, prod])
