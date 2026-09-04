"""Waterflooding while *under*-injecting -- and what it does to the front.

The "voidage replacement ratio" (VRR) is the injected volume divided by the
produced one. The incompressible model can only ever run `VRR = 1`
(`time_stepper` asserts it). With `ct > 0` we may inject less than we produce,
the difference being made up by expansion (storage), i.e. by declining pressure.

Here the same producer rate is run at `VRR = 1` and at `VRR = ½`, showing that
compressibility affects the *saturation* history, not just the pressure:

- The front advances more slowly, delaying water breakthrough.
- Some of the oil is instead driven by expansion, from all directions, so the
  sweep is not simply a "slowed down" version of `VRR = 1`: per unit of water
  injected, it is also *less efficient*.
- The pressure declines at the rate given by material balance
  (see `examples.depletion`), which is the price paid for the deferral.

Both the pressure and the transport equation carry their $O(c_t)$ terms
(ref `ResSim.ct`), so the saturations stay bounded and, at `VRR = 1`, converge
to the incompressible solution as $c_t → 0$ -- verified below.

.. warning:: This example is deliberately extreme.

    It produces two pore volumes while injecting one,
    so the fluids are asked to expand by 100%, i.e.
    $ c_t Δ\\bar{p} = 1 $, rather than the $ \\ll 1 $ that "slightly
    compressible" implies. The behaviour below is self-consistent and
    qualitatively right, but not quantitatively trustworthy. NB: lowering `ct`
    does *not* remedy it, the voidage being what sets the demanded expansion
    (ref the "Compressibility" section of the docs).

In the figures:

- "saturation": the VRR = ½ row lags from the outset (only half the water has
  gone in), and by t = 8 it has still not swept the NE corner, whereas VRR = 1
  has (mean saturation 0.79, against 0.89). Note also the *shape*: compared at
  equal injected volume (VRR = 1 at t = 4 against VRR = ½ at t = 8, both one
  pore volume) the two sweeps are close, but not equal -- an rms difference of
  0.06 in saturation.
- "histories" (left): breakthrough (dotted) is deferred from t = 2.8 to t = 3.9.
  Measured in injected volume, though, it arrives *sooner*: after 0.49 pore
  volumes, against 0.69. Under-injecting buys time, but sweeps less
  efficiently, the expansion drive adding a drift towards the producer.
- "histories" (right): the price. $\\bar{p}$ falls from 15 to 5, along the
  material-balance line (dashed), while VRR = 1 holds it at 15.
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup
q = .25
ct = .1
dt = .04
nSteps = 200
tt = dt*np.arange(nSteps + 1)
oil_only = np.zeros(32*32)


def waterflood(vrr, ct=ct, P0=15.):
    """Produce at rate `q`, inject at `vrr*q`."""
    model = ResSim(Lx=1, Ly=1, Nx=32, Ny=32, ct=ct,
                   wells=[dict(xy=[0, 0], rate=+vrr*q),
                          dict(xy=[1, 1], rate=-q)])
    kwargs: dict = dict(P0=P0*np.ones(model.Nxy)) if ct else {}
    return (model,) + model.sim(dt, nSteps, oil_only, pbar=False, **kwargs)


## Simulate
model, SS_full, PP_full = waterflood(vrr=1)
_    , SS_half, PP_half = waterflood(vrr=.5)
# At VRR = 1 there is no net voidage, so the incompressible model is the
# `ct → 0` limit. Check that the deviation from it is indeed O(ct), i.e. that
# dividing `ct` by ten divides the deviation by (about) ten.
_, SS_inc , _ = waterflood(vrr=1, ct=0)
_, SS_lowc, _ = waterflood(vrr=1, ct=ct/10)
dev = [abs(S - SS_inc).max() for S in [SS_full, SS_lowc]]
assert 8 < dev[0]/dev[1] < 12, dev

iprd = model.xy2ind(*model.wells.xy[1])
breakthrough = [dt*np.argmax(S[:, iprd] > .01) for S in [SS_full, SS_half]]

## Plot: the front, at equal times
fig, axs = freshfig("Voidage replacement -- saturation", nrows=2, ncols=3,
                    sharex=True, sharey=True, figsize=(9, 6))
for row, (SS, vrr) in enumerate(zip([SS_full, SS_half], ["1", "½"])):
    for col, t in enumerate([2, 4, 8]):
        model.plt_field(axs[row, col], SS[int(t/dt)], "oil", wells=dict(size=.4),
                        colorbar=False, finalize=False, labels=False,
                        title=(f"t = {t}" if row == 0 else ""))
    axs[row, 0].set_ylabel(f"VRR = {vrr}\ny")
axs[1, 0].set_xlabel("x")
fig.tight_layout()

## Plot: breakthrough, and the pressure paid for it
fig, (ax1, ax2) = freshfig("Voidage replacement -- histories",
                           ncols=2, figsize=(10, 4))

for SS, vrr, t_bt in zip([SS_full, SS_half], ["1", "½"], breakthrough):
    h, = ax1.plot(tt, 1 - SS[:, iprd], label=f"VRR = {vrr}")
    ax1.axvline(t_bt, c=h.get_color(), ls=":", lw=1)
ax1.set(title="Oil saturation in the producer\n(dotted: water breakthrough)",
        xlabel="Time", ylabel="$1 - S$")
ax1.legend()

for PP, vrr in zip([PP_full, PP_half], ["1", "½"]):
    ax2.plot(tt, PP.mean(axis=1), label=f"VRR = {vrr}")
ax2.plot(tt, 15 - .5*q*tt/(ct*model.h2*model.por.sum()), "k--", lw=1,
         label="$p_0 - (1 - \\mathrm{VRR}) q t / (c_t V_p)$")
ax2.set(title="Mean pressure", xlabel="Time", ylabel="$\\bar{p}$")
ax2.legend(fontsize="small")
fig.tight_layout()

# Under-injecting defers breakthrough, but not for free:
assert breakthrough[1] > breakthrough[0]
assert PP_half[-1].mean() < PP_full[-1].mean()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(sat_full = SS_full[-1],
                  sat_half = SS_half[-1],
                  prd_full = SS_full[:, iprd],
                  prd_half = SS_half[:, iprd],
                  p_half   = PP_half.mean(axis=1))

if __name__ == "__main__":
    show()
