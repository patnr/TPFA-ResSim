"""The gradient of the water cut at one producer, at one time, wrt. the field of $\\log K$ and the BHP schedule.

A five-spot: one injector at the centre, at a fixed rate, and four producers in
the corners, on BHP control (so their rates are outcomes, and the flow splits
among them as the heterogeneous -- smoothed, log-normal -- permeability
dictates). The objective is the water cut at the NE producer at a time index
just after its breakthrough, while it is rising fast.
`TPFA_ResSim.tlm.adjoint` returns its gradient with respect to every cell's
$\\log K$ (and the initial state) *and* with respect to every producer's BHP
at every time step, at the cost of about one more simulation -- here each is
checked against a finite difference in a random direction, which agrees to
about 1e-6 (relative; the floor is set by the kinks of the discrete map, ref
the "What is differentiated" section of `TPFA_ResSim.tlm`).

The water cut at a producer, $ f_w(s) $ in its cell (the fraction of water in
what it produces), is a function of the saturation there alone, so the adjoint
is seeded by the single entry $ ∂J/∂s_k[i_\\mathrm{prd}] = f_w'(s) $ -- which
`TPFA_ResSim.fluids.Fluid.fractional_flow` supplies.

In the figure:

- Top left: the $\\log K$ field, with the wells.
- Top right: the water cut of each producer over time, the objective marked.
- Bottom left: the gradient wrt. $\\log K$. It is positive along the flow path
  from the injector to the NE producer -- more permeable rock there brings the
  water sooner -- and negative along the paths to the *other* producers, and
  around the NE path: more permeable rock there diverts the water from the NE
  well, or lets it sweep a wider area (later arrival at the well itself).
  Outside the drainage area of the injector-NE pair, it vanishes: what happens
  there has not yet had time to affect the NE well.
- Bottom right: the gradient wrt. the BHP of each producer, per time step
  (`tlm.Gradient.bhp`). *Lowering* the NE well's BHP draws more of the
  injected water its way, hence an earlier breakthrough and a higher water cut
  at the objective's time: its gradient is negative, throughout. The opposite
  corner's (SW) is positive throughout: drawing more to it starves the NE
  well. The neighbours' (NW, SE) are smaller and change sign over time:
  drawing more flow towards a neighbour reshapes the stream tube to NE as much
  as it starves it, and which effect wins depends on the heterogeneity and on
  where the front is. All of them vanish from the objective's time step on --
  a control cannot affect what came before it. Their sum over time is the
  gradient with respect to a constant-in-time BHP.

.. note:: The gradient is with respect to the *isotropic* $\\log K$.

    `model.K` holds both components, so the adjoint returns a gradient for
    each; as they are set equal here, the gradient of the single field is
    their sum, ref `tlm.Gradient.logK`.

.. note:: The well indices are held fixed, as in the adjoint.

    They are computed (from the permeability at the wells) once, when the
    wells are configured, and stored (`Wells.WI`); the finite differences
    below reuse them, as `tlm` assumes (ref its "What is differentiated").
"""

from dataclasses import replace

from mpl_tools.place import freshfig
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter as smooth

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show
from TPFA_ResSim.tlm import adjoint

rng = np.random.default_rng(1)  # Reproducibility (the values are regression tested)

## Model: a five-spot on heterogeneous permeability
grid: dict = dict(Lx=1, Ly=1, Nx=32, Ny=32)
logK = 3 * smooth(smooth(rng.standard_normal((grid["Nx"], grid["Ny"]))))
model = ResSim(**grid, K=np.exp(logK), wells=[  # isotropic: K broadcast to both components
    dict(xy=[.5, .5], rate=+1, name="inj"),
    dict(xy=[1 , 1 ], bhp=0, rw=1e-3, name="NE"),
    dict(xy=[0 , 1 ], bhp=0, rw=1e-3, name="NW"),
    dict(xy=[0 , 0 ], bhp=0, rw=1e-3, name="SW"),
    dict(xy=[1 , 0 ], bhp=0, rw=1e-3, name="SE"),
])

dt, nSteps = .05, 30
S0 = np.zeros(model.Nxy)
SS, PP = model.sim(dt, nSteps, S0, pbar=False)

## The objective: water cut at the NE producer at time `k`
producers = model.wells.names[1:]
prd = model.xy2ind(*model.wells.xy[1:].T)  # their cells


def water_cut(model, SS):
    """`(nSteps+1, nPrd)` water cut at each producer, for each stored time."""
    return np.array([model.fluid.fractional_flow(S)[prd] for S in SS])


fw = water_cut(model, SS)
well, k = 0, 12  # NE, just after breakthrough (fw ≈ .5)
J = fw[k, well]

## Its gradient, by the adjoint
dJ_dSS = np.zeros_like(SS)
dJ_dSS[k, prd[well]] = model.fluid.dfractional_flow(SS[k])[prd[well]]  # f_w'(s)
grad = adjoint(model, dt, SS, PP, dJ_dSS)
G = grad.logK.sum(0)  # isotropic ⇒ sum the components
G_bhp = grad.bhp      # (nComp, nSteps); zero for the (rate-controlled) injector


## Check: finite differences in random directions of log K and of the BHP schedule
def J_of(logK, bhp):
    """The objective of a model with `logK`, its producers at the schedule `bhp`."""
    m = ResSim(**grid, K=np.exp(logK), wells=replace(model.wells, bhp=bhp))  # same `WI`
    return water_cut(m, m.sim(dt, nSteps, S0, pbar=False)[0])[k, well]


eps = 1e-5
bhp = model.wells.bhp  # (nComp, 1): constant in time
d_logK = rng.standard_normal(model.shape)
d_bhp = np.where(np.isfinite(bhp), rng.standard_normal((model.nComp, nSteps)), 0)
fd_logK = (J_of(logK + eps*d_logK, bhp) - J_of(logK - eps*d_logK, bhp)) / (2*eps)
fd_bhp = (J_of(logK, bhp + eps*d_bhp) - J_of(logK, bhp - eps*d_bhp)) / (2*eps)
directional_logK = (G * d_logK).sum()
directional_bhp = (G_bhp * d_bhp).sum()
assert abs(fd_logK - directional_logK) < 1e-4 * abs(directional_logK), (fd_logK, directional_logK)
assert abs(fd_bhp - directional_bhp) < 1e-4 * abs(directional_bhp), (fd_bhp, directional_bhp)

## Plot
fig, axs = freshfig("Water-cut gradient", ncols=2, nrows=2, figsize=(10, 8),
                    gridspec_kw={'width_ratios': (1, 1.2)})

ax = axs[0, 0]
model.plt_field(ax, logK, title="$\\log K$", cmap="viridis", levels=17,
                wells="color", finalize=False)

tt = dt * np.arange(nSteps + 1)
ax = axs[0, 1]
for i, name in enumerate(producers):
    ax.plot(tt, fw[:, i], label=name, c=f"C{i}")
ax.plot(tt[k], J, "o", c="k", mfc="none", ms=10, zorder=3,
        label=f"objective: {producers[well]} @ t={tt[k]:.2f}")
ax.set(title="Water cut", xlabel="Time", ylabel="$f_w$", ylim=(-.02, 1))
ax.legend(loc="upper left")

ax = axs[1, 0]
# The few cells next to the wells dominate; clip the color scale (the cmap's
# `over`/`under` make `plt_field` extend the colorbar, rather than leave blanks).
m = np.percentile(abs(G), 98)
cmap = plt.get_cmap("RdBu_r")
cmap = cmap.with_extremes(over=cmap(1.0), under=cmap(0.0))
model.plt_field(ax, G, title="$∂J/∂\\log K$", cmap=cmap,
                levels=np.linspace(-m, m, 21), cticks=[-m, 0, m],
                wells="color", finalize=False)

ax = axs[1, 1]
for i, name in enumerate(producers):
    ax.step(tt[:-1], G_bhp[1 + i], where="post", label=name, c=f"C{i}")
ax.axvline(tt[k], c="k", ls=":", lw=1, label="objective's time")
ax.axhline(0, c="k", lw=.5)
ax.set(title="$∂J/∂p_\\mathrm{bh}$, per time step", xlabel="Time (of the control)",
       ylabel="$∂J/∂p_\\mathrm{bh}$")
ax.legend(loc="upper right")

fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(water_cut    = fw,
                  gradient     = G,
                  gradient_bhp = G_bhp,
                  directional  = [directional_logK, fd_logK, directional_bhp, fd_bhp])

if __name__ == "__main__":
    show()
