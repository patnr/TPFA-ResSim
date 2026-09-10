"""A logo: the saturation field of a reservoir shaped like a smiley, or a yin-yang.

No physics is illustrated here that the other examples do not; the goal is a
picture. But it does show how little a shape costs on the rectangular grid:
an outline is a boolean expression in the mesh coordinates, assigned to
`TPFA_ResSim.ResSim.active` (ref `examples.inactive_cells`), holes included,
and the wells and the aquifer are records like any other's.

- **The smiley**: a disc, the smile *cut out* of it (inactive). The eyes are
  the injectors -- one well, two completions -- whose water pools around them,
  and the nose the producer, drawing it down. Along the bottom lies an
  **aquifer** (ref `examples.aquifer`), whose water rises *around* the smile to
  meet the producer -- which is set to take more than is injected, the aquifer
  supplying the rest. Set `aquifer = False` to remove it: the rates are then
  balanced, so the picture can be compared with and without. The aquifer's
  contact is stroked in blue (`TPFA_ResSim.plotting.Plot2D.plt_faces`).
- **The yin-yang**: a disc, the S-curve through it -- two semicircles of half
  its radius -- a *barrier* of inactive cells, like the fault of
  `examples.inactive_cells`, except that it stops short of the rim at the
  bottom. The two dots are the wells: the injector in the upper one, the
  producer in the lower. So the injector's half floods, and the water can reach
  the other half only by way of the gap at the bottom -- where the front stands
  at the time shown. The water is made ten times as viscous as the oil, so that
  the displacement is piston-like (the Buckley-Leverett shock is then at nearly
  full water saturation, ref `examples.buckley_leverett`): the flooded half is
  uniformly teal, and the front is sharp. With equal viscosities the front is a
  rarefaction -- a spread of pale contours -- and the two halves contrast poorly.

The figures are the plain `plt_field` (oil saturation: water in teal, oil in
coral), stripped of axes, colorbar, and labels; the well markers remain (the
eyes and nose, the dots, are the markers). The stroke of the aquifer contact runs
along the boundary *faces* of the contact cells, so it has no width but its `lw`,
and the contours stop half a cell short of it; `cellwise=True` would instead paint
the cells flat, up to those faces, with a pixelated outline.
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show
from TPFA_ResSim.wells import boundary_faces

aquifer = True  # toggle: water beyond the bottom boundary, at pressure 1
p_aq = 1.

def make(outline, wells, y_aq=None, **kws):
    """A 64² unit square cut to `outline(X, Y)`, with `wells`, and the aquifer below `y_aq`."""
    model = ResSim(Lx=1, Ly=1, Nx=64, Ny=64, **kws)
    X, Y = model.mesh
    model.active = outline(X, Y)
    if aquifer and y_aq is not None:
        # The contact: the boundary cells (those with a face to an inactive cell
        # or off-grid) of the lower part of the outline.
        xy = np.column_stack([X[model.active], Y[model.active]])
        contact = xy[boundary_faces(model, xy).any(-1) & (xy[:, 1] < y_aq)]
        wells = wells + [dict(name="Aq", xy=contact, aquifer=True, bhp=p_aq)]
    model.wells = wells
    return model

q_aq = .5 if aquifer else 0  # the deficit of injection, which the aquifer covers

## The smiley
def smiley(X, Y):
    face = (X - .5)**2 + (Y - .5)**2 <= .46**2
    r, angle = np.hypot(X - .5, Y - .52), np.arctan2(Y - .52, X - .5)  # polar about the nose
    smile = (abs(r - .27) <= .03) & (-.75 * np.pi < angle) & (angle < -.25 * np.pi)
    return face & ~smile

model = make(smiley, y_aq=.18, wells=[
    dict(name="Nose", xy=[.5, .5], rate=-(1 + q_aq)),
    dict(name="Eyes", xy=[[.25, .62], [.75, .62]], rate=+1),  # one well, 2 completions
])
S0 = np.zeros(model.Nxy)
SS, PP = model.sim(.004, 55, S0, pbar=False)

# The aquifer supplies exactly the deficit (incompressible), and the inactive cells stay inert.
assert np.allclose(model.wells.rates_by_well.sum(0), 0)
if aquifer:
    assert np.allclose(model.wells.rates_by_well[-1], q_aq)
assert (SS[:, ~model.active.ravel()] == 0).all()

## The yin-yang
R, w, gap = .46, .012, .07  # radius; the barrier's half-width; the opening at the bottom

def yinyang(X, Y):
    disc = np.hypot(X - .5, Y - .5) <= R
    upper = (abs(np.hypot(X - .5, Y - .5 - R/2) - R/2) <= w) & (X >= .5) & (Y >= .5)
    lower = (abs(np.hypot(X - .5, Y - .5 + R/2) - R/2) <= w) & (X <= .5) & (Y <= .5)
    return disc & ~((upper | lower) & (Y > .5 - R + gap))

yy_model = make(yinyang, fluid=dict(vw=10), wells=[  # viscous water ⇒ a piston-like front
    dict(name="Inj", xy=[.5, .75], rate=+1),
    dict(name="Prd", xy=[.5, .25], rate=-1),
])
SS_yy, PP_yy = yy_model.sim(.005, 62, S0, pbar=False)

# The barrier holds: the water enters the producer's half from the bottom, so the
# upper part of that half (right of the upper semicircle) is still dry.
X, Y = yy_model.mesh
far = yy_model.active & (X > .5) & (Y > .5) & (np.hypot(X - .5, Y - .5 - R/2) > R/2)
assert SS_yy[-1].reshape(yy_model.shape)[far].max() < 1e-6

## Plot
for name, m, S in [("smiley", model, SS[-1]), ("yin-yang", yy_model, SS_yy[-1])]:
    fig, ax = freshfig(f"Logo ({name})", figsize=(5, 5))
    m.plt_field(ax, S, "oil", colorbar=False, labels=False, finalize=False,
                title="", wells=dict(exclude=["Aq"], size=1.8, text=False))
    if "Aq" in m.wells.names:
        m.plt_faces(ax, m.wells.xy[m.wells.group == m.wells.nWell - 1], color="darkblue", lw=8)
    ax.axis("off")
    fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(smiley  = SS[-1][model.active.ravel()],
                  yinyang = SS_yy[-1][yy_model.active.ravel()])

if __name__ == "__main__":
    show()
