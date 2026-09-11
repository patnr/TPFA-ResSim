"""A logo: the saturation field of a reservoir shaped like a smiley.

No physics is illustrated here that the other examples do not; the goal is a
picture. But it does show how little a shape costs on the rectangular grid:
an outline is a boolean expression in the mesh coordinates, assigned to
`minires.ResSim.active` (ref `examples.inactive_cells`), holes included,
and the wells and the aquifer are records like any other's.

The face is a disc, with the smile *cut out* of it (inactive). The eyes are
the injectors -- one well, two completions -- whose water pools around them,
and the nose the producer, drawing it down. Along the bottom lies an
**aquifer** (ref `examples.aquifer`), whose water rises *around* the smile to
meet the producer -- which is set to take more than is injected, the aquifer
supplying the rest. Set `aquifer = False` to remove it: the rates are then
balanced, so the picture can be compared with and without. The aquifer's
contact is stroked in blue (`minires.plotting.Plot2D.plt_faces`).

The figure is the plain `plt_field` (oil saturation: water in teal, oil in
coral), stripped of axes, colorbar, and labels; the well markers remain (the
eyes and nose are the markers) -- each coloured individually, by name
(and by completion, for the two eyes), with the little centre dot turned off
(`wells=dict(color=..., dot=False)`, ref `minires.plotting.Plot2D.well_scatter`). The stroke of the aquifer contact runs
along the boundary *faces* of the contact cells, so it has no width but its `lw`,
and the contours stop half a cell short of it; `cellwise=True` would instead paint
the cells flat, up to those faces, with a pixelated outline.
"""

from typing import Any

from mpl_tools.place import freshfig
import numpy as np
from minires import ResSim
from minires.plotting import show
from minires.wells import boundary_faces

aquifer = True  # toggle: water beyond the bottom boundary, at pressure 1
p_aq = 1.
q_aq = .5 if aquifer else 0  # the deficit of injection, which the aquifer covers
y_aq = .18  # the aquifer lies below this

## The outline: a 32² unit square cut to a disc, less the smile
model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32)
X, Y = model.mesh
face = (X - .5)**2 + (Y - .5)**2 <= .46**2
r, angle = np.hypot(X - .5, Y - .52), np.arctan2(Y - .52, X - .5)  # polar about the nose
smile = (abs(r - .27) <= .03) & (-.75 * np.pi < angle) & (angle < -.25 * np.pi)
model.active = face & ~smile

## The wells (`Any`, so that the type checker sees `model.wells` as the `Wells` that
## `__setattr__` makes of these records, rather than as the records)
wells: Any = [dict(name="Nose", xy=[.5, .5], rate=-(1 + q_aq)),
              dict(name="Eyes", xy=[[.25, .62], [.75, .62]], rate=+1)]  # 1 well, 2 compl.
if aquifer:
    # The contact: the boundary cells (those with a face to an inactive cell
    # or off-grid) of the lower part of the outline.
    xy = np.column_stack([X[model.active], Y[model.active]])
    contact = xy[boundary_faces(model, xy).any(-1) & (xy[:, 1] < y_aq)]
    wells = wells + [dict(name="Aq", xy=contact, aquifer=True, bhp=p_aq)]
model.wells = wells

S0 = np.zeros(model.Nxy)
SS, PP = model.sim(.004, 55, S0, pbar=False)

# The aquifer supplies exactly the deficit (incompressible), and the inactive cells stay inert.
assert np.allclose(model.wells.rates_by_well.sum(0), 0)
if aquifer:
    assert np.allclose(model.wells.rates_by_well[-1], q_aq)
assert (SS[:, ~model.active.ravel()] == 0).all()

## Plot
# A colour per marker: keyed by well name, and, the eyes being one well of two
# completions, a colour per *completion* for them (left eye first, ref
# `minires.wells.Wells.xy`).
fig, ax = freshfig("Logo", figsize=(5, 5))
model.plt_field(ax, SS[-1], "oil", colorbar=False, labels=False, finalize=False, title="",
                wells=dict(exclude=["Aq"], size=1.5, text=False, dot=False))
if "Aq" in model.wells.names:
    contact = model.wells.xy[model.wells.group == model.wells.nWell - 1]
    model.plt_faces(ax, contact, lw=12)
ax.axis("off")
fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(smiley = SS[-1][model.active.ravel()])

if __name__ == "__main__":
    show()
