"""Primary depletion: production *without* injection.

Impossible with `ct = 0`: incompressibility leaves nowhere for the produced
fluid to come from, so `time_stepper` asserts that the rates balance.
With `ct > 0` the deficit is instead drawn from *storage*, i.e. from the
expansion of the rock and fluids as the pressure drops.

Two flow regimes are visible:

- **Transient** (early): the pressure disturbance has not yet reached the
  (closed) boundaries, so the reservoir behaves as if it were infinite.
- **Boundary-dominated** (late): the disturbance has swept the whole domain, and
  the pressure profile is *frozen in shape* -- it simply subsides at the constant
  rate dictated by material balance,

  $$ dp̄/dt = -q / (c_t V_p) $$

  ($V_p$ = pore volume), and the drawdown $p̄ - p_\\mathrm{cell}$ is constant.

There is no water anywhere here (`S = 0` throughout), so this is effectively a
single-phase example. NB: the model requires at least one injector to be
configured, so we include an idle (zero-rate) one.

In the figures:

- "pressure": the depression cone grows (t = 0.003 → 0.025) and then stops
  changing shape, while the entire field subsides (t = 0.025 → 0.1).
  The colour scale is shared, so that late, uniform darkening *is* the decline.
- "decline" (left): $\\bar{p}$ falls exactly along the material-balance line
  (dashed), and the producer's *cell* pressure runs parallel to it, one drawdown
  below.
- "decline" (right): that drawdown (note the logarithmic time axis) grows
  throughout the transient, and then settles at 0.15 -- the transition
  occurring at about the time $r^2/η$ that the disturbance needs to reach the
  boundary (dotted).

Throughout, $p_\\mathrm{cell}$ is the pressure of the *cell* that holds the well,
not the pressure in the wellbore: it is an average over an area of $h^2$, so
refining the grid deepens it without limit (here, 0.15 at 32² but 0.18 at 64²).
Converting it to a bottom-hole pressure is the job of the *well model* -- set
`prd_WI` (ref `ResSim.peaceman_WI`) and read `model.actual_bhp` instead.
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup
q = .25
model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32, ct=.1,
               inj_xy=[[0, 0]] , inj_rates=[[0]],
               prd_xy=[[.5, .5]], prd_rates=[[q]])

dt = 1e-3
nSteps = 100
tt = dt*np.arange(nSteps + 1)
oil_only = np.zeros(model.Nxy)
p0 = np.ones(model.Nxy)

## Simulate
SS, PP = model.sim(dt, nSteps, oil_only, p0=p0, pbar=False)
assert SS.max() == 0, "No water is injected, so none should appear."

iw = model.xy2ind(*model.prd_xy[0])
p_mean = PP.mean(axis=1)
p_cell = PP[:, iw]
pore_volume = model.h2 * model.por.sum()

## Plot: the pressure field, subsiding
fig, axs = freshfig("Depletion -- pressure", ncols=3, sharex=True, sharey=True,
                    figsize=(9, 3.5))
kws = dict(levels=np.linspace(PP.min(), 1, 21), cmap="viridis",
           colorbar=False, finalize=False, wells=dict(size=.4))
for i, (ax, k) in enumerate(zip(axs, [3, 25, nSteps])):
    cc = model.plt_field(ax, PP[k], **kws, labels=(i == 0), title=f"t = {k*dt:.3f}")
fig.colorbar(cc, ax=axs, shrink=.6, label="p")
# Note how the *shape* stops changing, while the level keeps dropping.

## Plot: decline and drawdown
fig, (ax1, ax2) = freshfig("Depletion -- decline", ncols=2, figsize=(10, 4))

ax1.plot(tt, p_mean, label="Mean, $\\bar{p}$")
ax1.plot(tt, p_cell, label="Producer cell, $p_\\mathrm{cell}$")
ax1.plot(tt, 1 - q*tt/(model.ct*pore_volume), "k--", lw=1,
         label="$p_0 - q t / (c_t V_p)$")
ax1.set(title="Pressure decline", xlabel="Time", ylabel="p")
ax1.legend()

ax2.plot(tt[1:], (p_mean - p_cell)[1:], "-o", ms=3)
ax2.set(title="Drawdown, $\\bar{p} - p_\\mathrm{cell}$", xlabel="Time",
        xscale="log", ylabel="$\\Delta p$")
ax2.axhline((p_mean - p_cell)[-1], c="k", ls="--", lw=1,
            label="Boundary-dominated value")
ax2.axvline(.25/(1/model.ct), c="C1", ls=":", lw=1,
            label="$r^2/η$, $r$ = ½ (to the boundary)")
ax2.legend(fontsize="small")
fig.tight_layout()

# Material balance holds exactly (see also tests/test_compressible.py)
assert np.allclose(p_mean, 1 - q*tt/(model.ct*pore_volume))

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(p_mean = p_mean,
                  p_cell = p_cell,
                  p_last = PP[-1])

if __name__ == "__main__":
    show()
