"""Pressure buildup after shut-in -- a "well test".

The producer flows for a while, and is then shut in (rate set to 0).
The pressure then *builds up*, asymptotically towards the average pressure,
which is now constant since nothing enters or leaves the (closed) reservoir.

Everything here is a consequence of `ct > 0`:

- Monitor points further from the well respond later
  (and, after shut-in, keep declining for a while before turning around:
  they have not yet "heard" that the well was shut).
- The relaxation is gradual, i.e. it has a memory of the flow history --
  this is what makes well testing a viable inference method.

For contrast, we also run the incompressible model -- where a shut-in is
felt everywhere immediately, and completely: the pressure (which is then only
defined up to a constant, ref `examples/pressure_diffusion.py`) instantly
becomes uniform. NB: since `ct = 0` demands balanced rates, that run needs an
active injector, whose rate is switched off at the same time.

As in `examples/depletion.py`, no water is present, so `S = 0` throughout --
and, likewise, every pressure plotted here is a *cell* pressure, not a wellbore
one; ref that example's note on the well model.

In the figures:

- "time series" (left): the delay of the response with distance -- by the time
  the well (r = 0) has dropped by 0.15, the r = 0.5 point has barely moved.
  After the shut-in the near-well pressure recovers at once, whereas the
  distant points keep *declining* for a while before turning around: they have
  not yet heard of it. All then converge on $\\bar{p}$ (dashed), now constant.
- "time series" (right): for `ct > 0` the drawdown decays smoothly over the
  remainder of the run. For `ct = 0` it is a rectangle: rate on, rate off,
  and nothing in between.
- "pressure": the depression cone filling in -- the sharp, near-well part
  first, the broad remainder last.
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup
q = .25
nSteps = 100
kShut = 25  # Time index of shut-in
dt = 2e-4
tt = dt*np.arange(nSteps + 1)

schedule = np.where(np.arange(nSteps) < kShut, q, 0)
# Aside: feedback control (e.g. shut-in upon water breakthrough) would instead
# be implemented by overriding `ResSim.well_controls`.

model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32, ct=.1,
               inj_xy=[[0, 0]]  , inj_rates=[[0]],
               prd_xy=[[.5, .5]], prd_rates=[schedule])

# Incompressible analogue: the injector must match the producer at all times.
model_inc = ResSim(Lx=1, Ly=1, Nx=32, Ny=32,
                   inj_xy=[[0, 0]]  , inj_rates=[schedule],
                   prd_xy=[[.5, .5]], prd_rates=[schedule])

oil_only = np.zeros(model.Nxy)
P0 = np.ones(model.Nxy)

## Simulate
SS, PP = model.sim(dt, nSteps, oil_only, P0=P0, pbar=False)
_ , PP_inc = model_inc.sim(dt, nSteps, oil_only, pbar=False)

iw = model.xy2ind(*model.prd_xy[0])
p_mean = PP.mean(axis=1)

## Plot: monitor points, and the drawdown
fig, (ax1, ax2) = freshfig("Buildup -- time series", ncols=2, figsize=(10, 4))

for r in [0, .1, .25, .5]:
    i = model.xy2ind(.5 + r, .5)
    ax1.plot(tt, PP[:, i], label=f"r = {r}")  # r = 0 is the well's cell
ax1.plot(tt, p_mean, "k--", lw=1, label="Mean, $\\bar{p}$")
ax1.axvline(kShut*dt, c="k", lw=1, alpha=.4)
ax1.annotate("shut-in", (kShut*dt, PP.min()), fontsize="small",
             xytext=(4, 0), textcoords="offset points")
ax1.set(title="Pressure at increasing distance from the well",
        xlabel="Time", ylabel="p")
ax1.legend(fontsize="small")

drawdown     = p_mean - PP[:, iw]
drawdown_inc = PP_inc.mean(axis=1) - PP_inc[:, iw]
ax2.plot(tt[1:], drawdown[1:]    , label="$c_t > 0$")
ax2.plot(tt[1:], drawdown_inc[1:], label="$c_t = 0$")
ax2.axvline(kShut*dt, c="k", lw=1, alpha=.4)
ax2.set(title="Drawdown, $\\bar{p} - p_\\mathrm{cell}$", xlabel="Time",
        ylabel="$\\Delta p$")
ax2.legend()
for ax in (ax1, ax2):
    ax.set_xticks(np.linspace(0, dt*nSteps, 5))
fig.tight_layout()

## Plot: the depression cone filling in
fig, axs = freshfig("Buildup -- pressure", ncols=4, sharex=True, sharey=True,
                    figsize=(11, 3.2))
kws = dict(levels=np.linspace(PP.min(), 1, 21), cmap="viridis",
           colorbar=False, finalize=False, wells=dict(size=.4))
for i, (ax, k) in enumerate(zip(axs, [kShut, kShut + 2, kShut + 10, nSteps])):
    cc = model.plt_field(ax, PP[k], **kws, labels=(i == 0), title=f"t = {k*dt:.4f}")
fig.colorbar(cc, ax=axs, shrink=.5, label="p")

# After shut-in, the average pressure is constant (nothing enters or leaves) ...
assert np.allclose(p_mean[kShut:], p_mean[kShut])
# ... and the pressure equilibrates towards it: by the end of the run, the
# spread has decayed to less than 1% of what it was at shut-in.
assert np.ptp(PP[-1]) < .01 * np.ptp(PP[kShut])
# Whereas the incompressible model forgets everything in a single step:
assert np.allclose(PP_inc[kShut + 1:], 0)

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(p_cell  = PP[:, iw],
                  p_far   = PP[:, model.xy2ind(1, .5)],
                  p_mean  = p_mean,
                  p_final = PP[-1])

if __name__ == "__main__":
    show()
