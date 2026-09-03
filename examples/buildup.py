"""Pressure buildup after shut-in -- a "well test". **In metric units.**

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

## Units

Unlike the other examples -- which leave the units unspecified (default)
this one is posed in practical **metric** units: metres, days, bar, mD, cP,
by changing `ResSim.cdarcy` to $ C = 0.008527 $.

Moreover, a 2 km square of 100 mD rock
at 250 bar, produced at 20 m³/day/m, has a pressure diffusivity of

$$ η = C K λ / (φ c_t) ≈ 4.3 \\times 10^4 \\; \\mathrm{m^2/day} \\,, $$

so the well feels the boundary, 1 km away, after some $ r^2/η ≈ 23 $ days --
whence the 12-day flow period and 28-day buildup simulated here.

It also lets the run be *interpreted* the way a real well test is. During
radial transient flow the line-source solution gives

$$ p_i - p = \\frac{q μ}{4 π C K} \\left[ \\ln \\frac{4 η t}{e^γ r^2} \\right] \\,, $$

so the semilog derivative $ dp / d\\ln t $ **plateaus** at $ q μ / (4 π C K) $ --
the classic diagnostic of pressure-transient analysis. Being independent of
$ r $, the *cell* pressure will do (no well model needed), and reading $ K $ off
that plateau recovers the 100 mD that went in, to within 4%. The shortfall is
the time discretization, not the units: at `dt = .05` it is 1%.

.. note:: Nothing here knows that the numbers are metric.

    The axis labels below say "[bar]" because *this script* says so.
    `cdarcy` fixes the arithmetic, not the nomenclature.

In the figures:

- "time series" (left): the delay of the response with distance -- by the time
  the well (r = 0) has dropped by 12 bar, the r = 1000 m point has barely moved.
  After the shut-in the near-well pressure recovers at once, whereas the
  distant points keep *declining* for a while before turning around: they have
  not yet heard of it. All then converge on $\\bar{p}$ (dashed), now constant.
- "time series" (middle): for `ct > 0` the drawdown decays smoothly over the
  remainder of the run. For `ct = 0` it is a rectangle: rate on, rate off,
  and nothing in between.
- "time series" (right): the semilog derivative, and the plateau read off its
  minimum. It is a *shallow* minimum rather than a flat stretch, being squeezed
  from both sides: at early times the cell average has not yet resolved the
  transient, and at late times (dotted) the closed boundary ends the radial
  regime. Hence the 4%.
- "pressure": the depression cone filling in -- the sharp, near-well part
  first, the broad remainder last.
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup -- a 2 km square of 100 mD rock at 250 bar
L      = 2000  # m
N      = 64
q      = 20    # m²/day, i.e. m³/day per metre of thickness
perm   = 100   # mD
mu     = 1     # cP
por    = .2
ct     = 1e-4  # 1/bar
p_i    = 250   # bar
dt     = .2    # day
nSteps = 200
kShut  = 60    # Time index of shut-in (t = 12 day)
tt     = dt*np.arange(nSteps + 1)

schedule = np.where(np.arange(nSteps) < kShut, q, 0)
# Aside: feedback control (e.g. shut-in upon water breakthrough) would instead
# be implemented by overriding `ResSim.well_controls`.

# `cdarcy` for m/day/bar/mD/cP: the darcy itself (9.869233e-16 m²),
# expressed in the system -- i.e. 0.008527. Ref `ResSim.cdarcy`.
C = 86400 * 9.869233e-16 * 1e5 / 1e-3

grid = dict(Lx=L, Ly=L, Nx=N, Ny=N, cdarcy=C,
            K=perm, por=por*np.ones((N, N)), vw=mu, vo=mu)  # fmt: skip

model = ResSim(**grid, ct=ct,
               wells=[dict(name="P1", xy=[L/2, L/2], rate=-schedule)])  # fmt: skip

# Incompressible analogue: the injector must match the producer at all times.
model_inc = ResSim(**grid,
                   wells=[dict(name="I1", xy=[0, 0], rate=+schedule),
                          dict(name="P1", xy=[L/2, L/2], rate=-schedule)])  # fmt: skip

oil_only = np.zeros(model.Nxy)
P0 = np.full(model.Nxy, p_i)

eta = C*perm/mu/(por*ct)  # Diffusivity (λ = 1, there being no water)
unit_p, unit_t = " [bar]", " [day]"

## Simulate
SS, PP = model.sim(dt, nSteps, oil_only, P0=P0, pbar=False)
_ , PP_inc = model_inc.sim(dt, nSteps, oil_only, pbar=False)

iw = model.xy2ind(*model.wells.xy[0])
p_mean = PP.mean(axis=1)
p_cell = PP[:, iw]

## Well test: the permeability, read off the semilog derivative's plateau
kk = np.arange(2, kShut)  # NB: skip t = 0, whose log is -inf
dp_dlnt = -(p_cell[kk+1] - p_cell[kk-1]) / (np.log(tt[kk+1]) - np.log(tt[kk-1]))
plateau = dp_dlnt.min()
perm_est = q*mu / (4*np.pi*C*plateau)

## Plot: monitor points, the drawdown, and the well test
fig, (ax1, ax2, ax3) = freshfig("Buildup -- time series", ncols=3, figsize=(14, 4))

for r in [0, 200, 500, 1000]:
    i = model.xy2ind(L/2 + r, L/2)
    ax1.plot(tt, PP[:, i], label=f"r = {r} m")  # r = 0 is the well's cell
ax1.plot(tt, p_mean, "k--", lw=1, label="Mean, $\\bar{p}$")
ax1.axvline(kShut*dt, c="k", lw=1, alpha=.4)
ax1.annotate("shut-in", (kShut*dt, PP.min()), fontsize="small",
             xytext=(4, 0), textcoords="offset points")
ax1.set(title="Pressure at increasing distance from the well",
        xlabel=f"Time{unit_t}", ylabel=f"p{unit_p}")
ax1.legend(fontsize="small")

drawdown     = p_mean - p_cell
drawdown_inc = PP_inc.mean(axis=1) - PP_inc[:, iw]
ax2.plot(tt[1:], drawdown[1:]    , label="$c_t > 0$")
ax2.plot(tt[1:], drawdown_inc[1:], label="$c_t = 0$")
ax2.axvline(kShut*dt, c="k", lw=1, alpha=.4)
ax2.set(title="Drawdown, $\\bar{p} - p_\\mathrm{cell}$", xlabel=f"Time{unit_t}",
        ylabel=f"$\\Delta p${unit_p}")
ax2.legend()

ax3.plot(tt[kk], dp_dlnt, "-o", ms=3)
ax3.axhline(plateau, c="k", ls="--", lw=1,
            label=f"Plateau ⇒ K = {perm_est:.0f} mD")
ax3.axvline((L/2)**2/eta, c="C2", ls=":", lw=1, label="$r^2/η$ (boundary)")
ax3.set(title="Well test: $dp / d\\ln t$", xlabel=f"Time{unit_t}", xscale="log",
        ylabel=f"$dp/d\\ln t${unit_p}")
ax3.legend(fontsize="small")
fig.tight_layout()

## Plot: the depression cone filling in
fig, axs = freshfig("Buildup -- pressure", ncols=4, sharex=True, sharey=True,
                    figsize=(11, 3.2))
kws = dict(levels=np.linspace(PP.min(), p_i, 21), cmap="viridis",
           colorbar=False, finalize=False, wells=dict(size=.4))
for i, (ax, k) in enumerate(zip(axs, [kShut, kShut + 2, kShut + 10, nSteps])):
    cc = model.plt_field(ax, PP[k], **kws, labels=(i == 0),
                         title=f"t = {k*dt:.1f} day")
fig.colorbar(cc, ax=axs, shrink=.5, label=f"p{unit_p}")

# After shut-in, the average pressure is constant (nothing enters or leaves) ...
assert np.allclose(p_mean[kShut:], p_mean[kShut])
# ... and the pressure equilibrates towards it: by the end of the run, the
# spread has decayed to less than 1% of what it was at shut-in.
assert np.ptp(PP[-1]) < .01 * np.ptp(PP[kShut])
# Whereas the incompressible model forgets everything in a single step:
assert np.allclose(PP_inc[kShut + 1:], 0)
# The well test recovers the permeability that went in, to within 4%
assert abs(perm_est/perm - 1) < .04

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(p_cell   = p_cell[::10],
                  p_far    = PP[::10, model.xy2ind(L, L/2)],
                  p_mean   = p_mean[::10],
                  p_final  = PP[-1],
                  perm_est = perm_est)

if __name__ == "__main__":
    show()
