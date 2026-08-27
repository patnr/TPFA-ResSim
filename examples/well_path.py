"""Well *paths*: a well completed in many cells rather than one.

`ResSim.well_path` walks a polyline through the grid and returns the cells it
traverses, their well indices (`ResSim.peaceman_WI`, scaled by how much of each
cell is actually traversed), and `w`, the resulting split of the well's rate.
Several completions then act as a single well simply by *being* several wells:
`ResSim._set_Q` superimposes them.

Here a point injector in the SW corner is replaced by a "horizontal" one along
the whole west edge, at the same total rate, and the difference in sweep is the
point of the exercise.

The rate split, though, is a choice, and the two modes of control make it
differently:

- under **rate** control, `w` apportions the rate in proportion to the well
  index -- a static allocation, exact only if the completions see equal
  pressure;
- under **BHP** control, they instead share a single $ p_\\mathrm{bh} $, and each
  flows whatever its own drawdown implies -- so the allocation is *solved for*,
  and follows the pressure field as it evolves.

This is incompressible (`ct = 0`), so the BHP-controlled path must still inject
exactly what the producer takes -- it only gets to choose *where* along itself.
Note also that a BHP well anchors the otherwise pure-Neumann pressure equation.

In the figures:

- "sweep": the point injector drives one front out of the corner, symmetric
  about the diagonal (the mean saturation is 0.66 in *both* the NW and the SE
  quadrant). The path drives a broad one off the whole west edge, and thereby
  breaks that symmetry: it floods the NW hard (0.87) at the expense of the SE
  (0.33), and in fact sweeps *less* of the field overall (83% against 90% of
  cells contacted). Longer is not automatically better -- which is the sort of
  thing one drills a path to find out. The well markers trace the path.
- "allocation" (left): the static split `w` is uniform here, this path crossing
  each cell of the west column fully -- it knows only the geometry. The
  BHP-solved split is not: the toe of the well, being nearer the producer, takes
  some 45% more than the heel. That ratio is set mostly by the geometry too, and
  so barely drifts as the flood develops.
- "allocation" (right): oil saturation at the producer. The path breaks through
  at step 21, against the point injector's 27, and its water cut then climbs
  faster -- the near half of the reservoir having been flooded preferentially.
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup
rw = 1e-3
nSteps = 28
dt = .7/nSteps
tt = dt*(1 + np.arange(nSteps))

def waterflood(**injector):
    """A unit square, producing from the NE corner at a rate of 1."""
    model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64,
                   prd_xy=[[1, 1]], prd_rates=[[1]], **injector)
    SS, PP = model.sim(dt, nSteps, model.swc*np.ones(model.Nxy), pbar=False)
    return model, SS

# The path, and its discretization into completions
proto = ResSim(Lx=1, Ly=1, Nx=64, Ny=64)
xy, WI, w = proto.well_path([[0, 0], [0, 1]], rw)
assert len(xy) == proto.Ny, "The west edge is one cell wide, and Ny cells long."

## Simulate: a point injector, the same as a path, and the path on BHP
point, SS_point = waterflood(inj_xy=[[0, 0]], inj_rates=[[1]])
path , SS_path  = waterflood(inj_xy=xy, inj_rates=w[:, None], inj_WI=WI)
onbhp, SS_onbhp = waterflood(inj_xy=xy, inj_WI=WI,
                             inj_bhp=np.full((len(xy), 1), 3.))

# Incompressible => whatever the control, the injection must match the production
for model in [path, onbhp]:
    assert np.allclose(model.actual_rates["inj"].sum(axis=0), 1)

## Plot: the resulting sweeps
fig, axs = freshfig("Well path -- sweep", ncols=2, sharex=True, sharey=True,
                    figsize=(9, 4))
for ax, (name, model, SS) in zip(axs, [("Point injector", point, SS_point),
                                       ("Path injector", path, SS_path)]):
    model.plt_field(ax, SS[-1], "oil", finalize=False, colorbar=False,
                    title=f"{name}, t = {dt*nSteps:.2f}", wells=dict(size=.3))
fig.tight_layout()

## Plot: how the rate gets allocated along the path, and what is produced
fig, (ax1, ax2) = freshfig("Well path -- allocation", ncols=2, figsize=(10, 4))

yy = xy[:, 1]
ax1.plot(yy, path.actual_rates["inj"][:, -1], label="Rate-controlled: $w \\propto WI$")
for k, ls in [(0, ":"), (nSteps - 1, "-")]:
    ax1.plot(yy, onbhp.actual_rates["inj"][:, k], ls, c="C1",
             label=f"BHP-controlled, t = {tt[k]:.2f}")
ax1.set(title="Injection rate per completion", xlabel="y (along the path)",
        ylabel="q", ylim=0)
ax1.legend(fontsize="small")

prd = [point.xy2ind(*point.prd_xy[0])]
for name, SS in [("Point", SS_point), ("Path", SS_path), ("Path, on BHP", SS_onbhp)]:
    ax2.plot(tt, 1 - SS[1:, prd[0]], label=name)
ax2.set(title="Oil saturation at the producer", xlabel="Time", ylabel="1 - s")
ax2.legend()
fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(alloc_static = w,
                  alloc_bhp    = onbhp.actual_rates["inj"][:, -1],
                  sat_point    = SS_point[-1, ::600],
                  sat_path     = SS_path[-1, ::600])

if __name__ == "__main__":
    show()
