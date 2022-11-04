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

from simulator import ResSim, recurse

plt.ion()

model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64)
model.config_wells(inj =[[0, 0, 1]],
                   prod=[[1, 1, -1]])

## Run
water_sat0 = np.zeros(model.M)
nSteps = 28
dt = 0.7/nSteps
SS = recurse(model.time_stepper(dt), nSteps, water_sat0)
# Final+1 pressure
[P, V] = model.pressure_step(SS[-1])


## Plot
kws = dict(levels=17, cmap="jet", origin=None,
           extent=(0, model.Lx, 0, model.Ly))

fig, axs = freshfig("Fig. 6", nrows=2, ncols=3, sharex=True, sharey=True,
                    subplot_kw={'aspect': 'equal'})

for ax, t in zip(axs.ravel(), [None, .14, .28, .42, .56, .70]):
    if ax.get_subplotspec().is_last_row(): ax.set_xlabel("x")  # noqa
    if ax.get_subplotspec().is_first_col(): ax.set_ylabel("y")  # noqa

    if t is None:
        ax.set_title("Pressure")
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


## Tests
# Automatically run by pytest

def test_compare_final_output():
    # From matlab_codes/listing9.m: >> S(1:100:end)
    matlab_sim_output = np.array([
        9.9965336e-01, 8.6608581e-01, 9.8617568e-01, 8.1474847e-01, 9.5755343e-01,
        7.4564308e-01, 9.1967705e-01, 2.2022926e-03, 8.7671705e-01, 9.6753659e-01,
        8.2997699e-01, 9.4122839e-01, 7.7584288e-01, 9.0680721e-01, 6.5725651e-01,
        8.6872144e-01, 9.2452225e-01, 8.2920328e-01, 9.0268120e-01, 7.8730324e-01,
        8.7388545e-01, 7.3151827e-01, 8.4228255e-01, 6.1365262e-04, 8.1067867e-01,
        8.4694041e-01, 7.8003369e-01, 8.2184061e-01, 7.4772389e-01, 7.9410484e-01,
        6.7467063e-01, 7.6588654e-01, 7.6829372e-01, 7.3788123e-01, 7.3259091e-01,
        7.0842952e-01, 6.6065423e-01, 6.6872765e-01, 6.8125425e-02, 5.8761446e-01,
        9.1089927e-20,
    ])
    assert np.all(np.isclose(SS[-1, ::100], matlab_sim_output))


## Animation
if __name__ == "__main__":
    from simulator import plotting
    plotting.model = model
    plotting.coord_type = "absolute"
    prod = [model.xy2ind(x, y) for (x, y, _) in model.producers]
    animation = plotting.anim(None, SS, SS[1:, prod])
