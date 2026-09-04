"""The gradient of the water cut at one producer, at one time, wrt. the field of $\\log K$.

A five-spot: one injector at the centre, four producers in the corners, on a
heterogeneous (smoothed, log-normal) permeability. The objective is the water
cut at the NE producer at a time index just after its breakthrough, where it is
rising fastest. `TPFA_ResSim.tlm.adjoint` returns its gradient with respect to
every cell's $\\log K$ (and the initial state) at the cost of about one more
simulation -- here checked against a finite difference in a random direction,
which agrees to about 1e-8.

The water cut at a producer, $ f_w(s) $ in its cell (the fraction of water in
what it produces; at a fixed rate, proportional to its water rate), is a
function of the saturation there alone, so the adjoint is seeded by the single
entry $ ∂J/∂s_k[i_\\mathrm{prd}] = f_w'(s) $ -- which `tlm.fractional_flow`
supplies.

In the figure:

- Left: the $\\log K$ field, with the wells.
- Middle: the water cut of each producer over time, the objective marked.
- Right: the gradient. It is positive along the flow path from the injector
  to the NE producer -- more permeable rock there brings the water sooner --
  and negative along the paths to the *other* producers, and around the NE
  path: more permeable rock there diverts the water from the NE well, or lets
  it sweep a wider area (later arrival at the well itself). Outside the
  drainage area of the injector-NE pair, it vanishes: what happens there has
  not yet had time to affect the NE well.

.. note:: The gradient is with respect to the *isotropic* $\\log K$.

    `model.K` holds both components, so the adjoint returns a gradient for
    each; as they are set equal here, the gradient of the single field is
    their sum, ref `tlm.Gradient.logK`.
"""

from mpl_tools.place import freshfig
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter as smooth

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show
from TPFA_ResSim.tlm import adjoint, fractional_flow

rng = np.random.default_rng(1)  # Reproducibility (the values are regression tested)

## Model: a five-spot on heterogeneous permeability
model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32, wells=[
    dict(xy=[.5, .5], rate=+1  , name="inj"),
    dict(xy=[1 , 1 ], rate=-.25, name="NE"),
    dict(xy=[0 , 1 ], rate=-.25, name="NW"),
    dict(xy=[0 , 0 ], rate=-.25, name="SW"),
    dict(xy=[1 , 0 ], rate=-.25, name="SE"),
])
logK = 3 * smooth(smooth(rng.standard_normal(model.shape)))
model.K = np.exp(logK)  # isotropic: broadcast to both components

dt, nSteps = .05, 30
S0 = np.zeros(model.Nxy)
SS, PP = model.sim(dt, nSteps, S0, pbar=False)

## The objective: water cut at the NE producer at time `k`
producers = model.wells.names[1:]
prd = model.xy2ind(*model.wells.xy[1:].T)  # their cells


def water_cut(model, SS):
    """`(nSteps+1, nPrd)` water cut at each producer, for each stored time."""
    return np.array([fractional_flow(model, S)[0][prd] for S in SS])


fw = water_cut(model, SS)
well, k = 0, 15  # NE, just after breakthrough (fw ≈ .5)
J = fw[k, well]

## Its gradient, by the adjoint
dJ_dSS = np.zeros_like(SS)
dJ_dSS[k, prd[well]] = fractional_flow(model, SS[k])[1][prd[well]]  # f_w'(s)
grad = adjoint(model, dt, SS, PP, dJ_dSS)
G = grad.logK.sum(0)  # isotropic ⇒ sum the components

## Check: a finite difference in a random direction of log K
direction = rng.standard_normal(model.shape)
eps = 1e-5
def J_of(logK):  # noqa: E302
    m = ResSim(Lx=1, Ly=1, Nx=32, Ny=32, wells=model.wells)
    m.K = np.exp(logK)
    return water_cut(m, m.sim(dt, nSteps, S0, pbar=False)[0])[k, well]
fd = (J_of(logK + eps*direction) - J_of(logK - eps*direction)) / (2*eps)
directional = (G * direction).sum()
assert abs(fd - directional) < 1e-4 * abs(directional), (fd, directional)

## Plot
fig, axs = freshfig("Water-cut gradient", ncols=3, figsize=(13, 4),
                    gridspec_kw={'width_ratios': (1, 1.2, 1)})

ax = axs[0]
model.plt_field(ax, logK, title="$\\log K$", cmap="viridis", levels=17,
                wells="color", finalize=False)

ax = axs[1]
tt = dt * np.arange(nSteps + 1)
for i, name in enumerate(producers):
    ax.plot(tt, fw[:, i], label=name, c=f"C{i}")
ax.plot(tt[k], J, "o", c="k", mfc="none", ms=10, zorder=3,
        label=f"objective: {producers[well]} @ t={tt[k]:.2f}")
ax.set(title="Water cut", xlabel="Time", ylabel="$f_w$", ylim=(-.02, 1))
ax.legend(loc="upper left")

ax = axs[2]
# The few cells next to the wells dominate; clip the color scale (the cmap's
# `over`/`under` make `plt_field` extend the colorbar, rather than leave blanks).
m = np.percentile(abs(G), 98)
cmap = plt.get_cmap("RdBu_r")
cmap = cmap.with_extremes(over=cmap(1.0), under=cmap(0.0))
model.plt_field(ax, G, title="$∂J/∂\\log K$", cmap=cmap,
                levels=np.linspace(-m, m, 21), cticks=[-m, 0, m],
                wells="color", finalize=False)

fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(water_cut   = fw,
                  gradient    = G,
                  directional = [directional, fd])

if __name__ == "__main__":
    show()
