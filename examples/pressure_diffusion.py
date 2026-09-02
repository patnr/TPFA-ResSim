"""Finite-speed pressure propagation, and the pressure "gauge".

With `ct = 0` the pressure equation is *elliptic*: the whole field re-adjusts
**instantaneously** whenever a rate changes, and (since the boundaries are closed)
the solution is only determined up to an additive constant.

With `ct > 0` it is *parabolic* -- a diffusion equation with diffusivity

$$ η = K λ / (φ c_t) $$

so a rate change is felt only after a delay of the order $ r^2/η $ at distance $r$.
The elliptic solution is recovered as $ t → ∞ $ (or $ c_t → 0 $), as shown below.
Moreover, the pressure level is now pinned by the initial condition `P0`
(the model keeps track of *absolute* pressure, not just its gradients).

Notes on the setup:

- The reservoir is initialised **fully water-saturated**, so that the total mobility
  $λ = 1$ is uniform and constant, isolating the pressure physics
  (i.e. $η = 100$ everywhere, since `K = por = 1` and `ct = .01`).
  It stays that way *exactly*: since the storage is charged to the phases in
  proportion to their saturation (ref `ResSim.ct`), $s = 1$ is a fixed point of
  the transport step, whatever the wells do. Asserted below.
- The rates are balanced (as they must be for the `ct = 0` comparison run),
  whence the storage terms cancel and the *mean* pressure stays at `P0` exactly.
  This conveniently fixes the (otherwise arbitrary) level of the elliptic solution:
  we centre it on its own mean.

In the figures:

- "Pressure diffusion": at t = 0.0002 only the immediate surroundings of the
  two wells have responded -- the middle of the domain is still at `P0` (white)
  -- while by t = 0.005 the field is barely distinguishable from the elliptic
  one (bottom right). The four panels share their colour scale.
- "profiles & gauge" (left): the same, quantified along the y = 0 edge. At the
  earliest time, the far half of that line has attained under 2% of its
  eventual (elliptic) response; at the latest time, some 98%. The dotted
  verticals mark the diffusion length $\\sqrt{ηt}$: an indication of scale,
  not a sharp front.
- "profiles & gauge" (right): the two `ct > 0` curves are separated by exactly
  the difference of their `P0` (asserted below) -- the absolute level is
  meaningful, and remembered. The two `ct = 0` curves instead coincide, at 0:
  `P0` is ignored, and the level is merely that of the grounded cell.
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup
wells = [dict(xy=[0, 0], rate=+.25), dict(xy=[1, 1], rate=-.25)]
grid = dict(Lx=1, Ly=1, Nx=32, Ny=32)

model = ResSim(**grid, wells=wells, ct=.01)
model_inc = ResSim(**grid, wells=wells)  # ct = 0

eta = 1/model.ct  # Diffusivity (since K = por = λ = 1)
dt = 2e-4
nSteps = 25
water_sat0 = np.ones(model.Nxy)
P0 = np.ones(model.Nxy)

## Simulate
SS, PP = model.sim(dt, nSteps, water_sat0, P0=P0, pbar=False)
assert (SS == 1).all(), "The reservoir should remain fully water-saturated."
_, PP_inc = model_inc.sim(dt, nSteps, water_sat0, pbar=False)
# Same, but starting from a higher pressure level
_, PP_hi = model.sim(dt, nSteps, water_sat0, P0=P0 + 1, pbar=False)
_, PP_inc_hi = model_inc.sim(dt, nSteps, water_sat0, P0=P0 + 1, pbar=False)

# The elliptic solution is the same at all times here (mobility is constant).
# Its level is arbitrary: centre it, matching the (conserved) mean of `PP`.
elliptic = PP_inc[1] - PP_inc[1].mean()

## Plot: the pressure disturbance, dp, spreading out
snapshots = [1, 5, nSteps]
vmax = 1.05*np.abs(elliptic).max()
kws = dict(levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r",
           colorbar=False, finalize=False, wells=dict(size=.4))

fig, axs = freshfig("Pressure diffusion", nrows=2, ncols=2,
                    sharex=True, sharey=True, figsize=(7, 6))
for ax, k in zip(axs.ravel(), snapshots + [None]):
    if k is None:
        cc = model.plt_field(ax, elliptic, **kws, title="$c_t = 0$: instant")
    else:
        cc = model.plt_field(ax, PP[k] - P0, **kws,
                             title=f"t = {k*dt:.4f}   "
                                   f"($\\sqrt{{ηt}}$ = {np.sqrt(eta*k*dt):.2f})")
    ax.title.set_fontsize("medium")
fig.colorbar(cc, ax=axs, shrink=.6, label="$p - p_0$")

## Plot: how much of the eventual response has arrived, and where
fig, (ax1, ax2) = freshfig("Pressure diffusion -- profiles & gauge",
                           ncols=2, figsize=(10, 4))

# Sample along the y=0 edge, stopping short of the (anti-symmetric) corner,
# where the elliptic reference vanishes and the ratio below is meaningless.
xx = np.linspace(0, .75, 13)
line = [model.xy2ind(x, 0) for x in xx]

for k in [1, 3, 9, nSteps]:
    h, = ax1.plot(xx, ((PP[k] - P0)/elliptic)[line], "-o", ms=3,
                  label=f"t = {k*dt:.4f}")
    # The diffusion length -- an indication of scale, not a sharp front
    ax1.axvline(np.sqrt(eta*k*dt), c=h.get_color(), ls=":", lw=1)
ax1.axhline(1, c="k", ls="--", lw=1, label="$c_t = 0$ (elliptic)")
ax1.set(title="Fraction of the elliptic response attained\n"
              "(along $y=0$; dotted: diffusion length $\\sqrt{ηt}$)",
        xlabel="x", ylabel="$(p - p_0) \\, / \\, p_\\mathrm{elliptic}$")
ax1.legend(fontsize="small")

# Gauge: is the absolute pressure level meaningful?
tt = dt*np.arange(nSteps + 1)
iw = model.xy2ind(*model.wells.xy[0])
ax2.plot(tt, PP[:, iw]   , "-" , c="C0", label="$c_t>0$, $p_0=1$")
ax2.plot(tt, PP_hi[:, iw], "--", c="C1", label="$c_t>0$, $p_0=2$")
ax2.plot(tt[1:], PP_inc[1:, iw]   , "-" , c="C2", label="$c_t=0$, $p_0=1$")
ax2.plot(tt[1:], PP_inc_hi[1:, iw], "--", c="C3", label="$c_t=0$, $p_0=2$")
ax2.set(title="Pressure at the injector", xlabel="Time", ylabel="p")
ax2.legend(fontsize="small")
fig.tight_layout()

# With ct > 0 the two curves are offset by exactly the offset in `P0`:
assert np.allclose(PP_hi - PP, 1)
# With ct = 0, `P0` is simply ignored, and the level is that of the "grounding"
# of cell 0 (ref article p. 13). Indeed, summing all rows of that (modified)
# system leaves `2 K p[0] = sum(Q) = 0`, i.e. it pins the pressure of cell (0,0):
assert np.allclose(PP_inc[1:], PP_inc_hi[1:])
assert np.allclose(PP_inc[1:, 0], 0)

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(dp_early = PP[1] - P0,
                  dp_late  = PP[nSteps] - P0,
                  elliptic = elliptic,
                  p_inj    = PP[:, iw])

if __name__ == "__main__":
    show()
