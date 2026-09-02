"""Rate control, BHP control, and the well model that connects them.

A well can be told *what to do* in two ways: give it a rate (`rate`, ref
`ResSim.well_rates`; negative to produce) and let its pressure follow, or give
it a pressure (`bhp`, ref `ResSim.well_bhp`) and let its rate follow -- as
`ResSim.wells` accepts either. Both need a **well model** -- the well index
(`ResSim.peaceman_WI`, whence the `rw` of the record) -- because a well is far
smaller than the cell that holds it, so its cell pressure is not its wellbore
pressure.

That distinction is the first thing shown here: the cell pressure is a *grid
artefact* (an average over an area of $h^2$, which deepens without limit as the
grid is refined), whereas the bottom-hole pressure obtained from it is not.

The two modes are then contrasted on the same closed, depleting reservoir
(cf. `depletion.py`), where they behave quite differently:

- at constant **rate**, the pressure declines linearly, as material balance
  dictates: $ d\\bar{p}/dt = -q / (c_t V_p) $;
- at constant **BHP**, the rate declines *exponentially*,
  $ q ∝ e^{-t/τ} $ with $ τ = c_t V_p / J $, since combining that same material
  balance with the well model, $ q = J (\\bar{p} - p_\\mathrm{bh}) $, gives a
  linear ODE. (The productivity index $ J $ differs from $ WI λ_t $ by the
  geometry between the well and the average pressure.)

Neither mode alone is how a well is actually run: the industry standard is a
rate *target* with a BHP *limit*, i.e. whichever of the two currently binds.
The model does not switch modes natively (a BHP well simply flows whichever
way its `p_bh` vs. cell pressure dictates), but `ResSim.well_controls`
returns both controls, so an override can switch between them -- lagged by the
step whose pressure it must judge from. That is the third case shown here.

Finally, the two are shown to be one and the same model, seen from either end:
prescribing the BHP that the rate-controlled run *reported* recovers that run
exactly (to ~1e-15, since the well model is solved simultaneously with the
pressure field, not lagged by a step).

In the figures:

- "diagnostic": refining 32² → 64² moves the producer's cell pressure by a lot
  (left), while the bottom-hole pressure inferred from it barely moves (right).
  Only the latter is a property of the *well*.
- "modes" (left): the rate is flat by construction under rate control, and
  decays under BHP control -- as a straight line on the log axis, i.e.
  exponentially, with the analytic slope (dashed). The rate-with-a-limit case
  traces the former until the limit binds, and joins the latter thereafter
  (above it, having drained less by then, hence at a higher pressure).
- "modes" (right): the mirror image. The BHP falls linearly under rate control
  (material balance), and is flat by construction under BHP control -- while the
  limited well does both in turn, its corner marking the switch.
- "duality": feeding the left run's BHP back in as the control reproduces its
  rate to machine precision.
"""

from mpl_tools.place import freshfig
import numpy as np

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup
q = .25          # the rate-controlled rate
p_bh = .5        # the BHP-controlled pressure
rw = 1e-3        # well radius
ct = .1
dt, nSteps = 2e-3, 150
tt = dt*np.arange(1, nSteps + 1)

def depleter(N=32, cls=ResSim, **control):
    """A single producer at the centre of a closed square. Cf. `depletion.py`.

    The `control` is a `rate` and/or a `bhp`; `rw` is what gives it a well model.
    """
    return cls(Lx=1, Ly=1, Nx=N, Ny=N, ct=ct,
               wells=[dict(xy=[.5, .5], rw=rw, **control)])

class Limited(ResSim):
    """Rate control with a BHP limit, by overriding `ResSim.well_controls`.

    The rate target is held for as long as it can be delivered without drawing
    the well below `p_bh`; thereafter the well switches to BHP control at it.
    The switch is judged from the *previous* step's pressure, since the new one
    is not yet known (indeed it depends on the choice) -- so the limit is
    breached for the one step in which it comes to bind.
    """

    def well_controls(self, S, P, k):
        ctrl = super().well_controls(S, P, k)
        if P is None:
            return ctrl                                # nothing to switch on
        would = self.bhp(S, P, ctrl["rates"])          # if rate-controlled
        ctrl["bhp"] = np.where(would < p_bh, p_bh, np.nan)
        return ctrl

def run(model):
    SS, PP = model.sim(dt, nSteps, np.zeros(model.Nxy),
                       P0=np.ones(model.Nxy), pbar=False)
    assert SS.max() == 0, "No water is injected, so none should appear."
    return PP

## Plot: the diagnostic -- cell pressure is a grid artefact, bottom-hole is not
fig, (ax1, ax2) = freshfig("Well control -- diagnostic", ncols=2, figsize=(10, 4),
                           sharey=True)
for N in [32, 64]:
    model = depleter(N, rate=-q)
    PP = run(model)
    pbar = PP.mean(axis=1)
    ax1.plot(tt, (pbar[1:] - PP[1:, model.xy2ind(*model.well_xy[0])]), label=f"{N}²")
    ax2.plot(tt, (pbar[1:] - model.actual_bhp[0]), label=f"{N}²")
ax1.set(title="Cell drawdown, $\\bar{p} - p_\\mathrm{cell}$",
        xlabel="Time", ylabel="$\\Delta p$")
ax2.set(title="Bottom-hole drawdown, $\\bar{p} - p_\\mathrm{bh}$", xlabel="Time")
for ax in (ax1, ax2):
    ax.legend(title="Grid")
fig.tight_layout()

## Simulate: the same reservoir, under either mode of control
by_rate = depleter(rate=-q)
PP_rate = run(by_rate)
by_bhp = depleter(bhp=p_bh)              # NB: no `rate` given
PP_bhp = run(by_bhp)
limited = depleter(cls=Limited, rate=-q)  # ... rate, but limited by p_bh
run(limited)                              # (its rate/BHP is the interest)

# Production, i.e. the negated (signed) rates
prod_rate, prod_bhp, prod_lim = [-m.actual_rates[0]
                                 for m in [by_rate, by_bhp, limited]]

# Analytic decline: q = J (pbar - p_bh) with material balance ct Vp dpbar/dt = -q
Vp = by_bhp.h2 * by_bhp.por.sum()
J = prod_bhp[-1] / (PP_bhp[-1].mean() - p_bh)
tau = ct*Vp/J

## Plot: rate and BHP, under either mode
fig, (ax1, ax2) = freshfig("Well control -- modes", ncols=2, figsize=(10, 4))

ax1.plot(tt, prod_rate, label="Rate-controlled")
ax1.plot(tt, prod_bhp, label="BHP-controlled")
ax1.plot(tt, prod_lim, ":", lw=2, label="Rate, limited")
ax1.plot(tt, prod_bhp[-1]*np.exp((tt[-1] - tt)/tau), "k--",
         lw=1, label=f"$\\propto e^{{-t/\\tau}}$, $\\tau = c_t V_p / J$ = {tau:.3f}")
ax1.set(title="Production rate", xlabel="Time", ylabel="q", yscale="log")
ax1.legend(fontsize="small")

ax2.plot(tt, by_rate.actual_bhp[0], label="Rate-controlled")
ax2.plot(tt, by_bhp .actual_bhp[0], label="BHP-controlled")
ax2.plot(tt, limited.actual_bhp[0], ":", lw=2, label="Rate, limited")
ax2.plot(tt, 1 - q*tt/(ct*Vp) - (PP_rate[-1].mean() - by_rate.actual_bhp[0, -1]),
         "k--", lw=1, label="$p_0 - qt/(c_t V_p) - \\Delta p$")
ax2.set(title="Bottom-hole pressure", xlabel="Time", ylabel="$p_\\mathrm{bh}$")
ax2.legend(fontsize="small")
fig.tight_layout()

## The duality: prescribe the BHP that the rate-controlled run reported
replay = depleter(bhp=by_rate.actual_bhp[0])
PP_replay = run(replay)
err_P = np.abs(PP_replay - PP_rate).max()
err_q = np.abs(replay.actual_rates + q).max()
assert err_P < 1e-12 and err_q < 1e-12, "The two controls are not each other's inverse!"

fig, ax = freshfig("Well control -- duality", figsize=(6, 4))
ax.plot(tt, prod_rate, lw=4, alpha=.4, label="Rate-controlled: $q$")
ax.plot(tt, -replay.actual_rates[0], "k--", lw=1,
        label="BHP-controlled by its own reported $p_\\mathrm{bh}$")
ax.set(title=f"The same well, controlled from either end (max err {err_q:.0e})",
       xlabel="Time", ylabel="q", ylim=(0, 2*q))
ax.legend()
fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
# NB: the production rates are negated, preserving the pre-v0.3 references.
__digest__ = dict(rate_of_bhp_ctrl = prod_bhp,
                  bhp_of_rate_ctrl = by_rate.actual_bhp[0],
                  p_last_bhp_ctrl  = PP_bhp[-1],
                  rate_of_limited  = prod_lim,
                  bhp_of_limited   = limited.actual_bhp[0])

if __name__ == "__main__":
    show()
