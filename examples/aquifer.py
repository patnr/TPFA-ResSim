"""An aquifer: water beyond the boundary, feeding the reservoir at its own pressure.

Beyond some stretch of the reservoir's boundary there may lie water-bearing
rock -- an *aquifer* -- at a pressure of its own, which then feeds the
reservoir cells touching it at a rate proportional to the pressure difference
across the boundary face. That is the law of a BHP-controlled well, so an
aquifer *is* one: a "well" completed in every contact cell, with the aquifer
pressure for `bhp` and, for `WI`, the transmissibility of the boundary face(s)
of the cell -- `minires.wells.aquifer_WI`, which the record key
`aquifer=True` applies for you. Nothing else of the model is involved: the
influx enters the equations like a well's, anchors the pressure (so that even
an incompressible reservoir needs no injector), and is reported like a well's
(`minires.wells.Wells.actual_rates`).

Here an elliptic reservoir (ref `examples.inactive_cells`) has an aquifer
along its western end, and a single producer to the east, on BHP control,
so that the rate is left to follow the pressures. Two aquifer models:

- **Constant pressure**, i.e. an infinite aquifer: the influx settles to the
  rate the mobilities admit, and the front sweeps the oil towards the producer.
- **Fetkovich**, i.e. a finite one: its pressure falls in proportion to the
  cumulative influx, $ p_\\mathrm{aq} = p_i (1 - W_e / W_{ei}) $, where the
  capacity $ W_{ei} $ is the aquifer's compressibility times its volume times
  $ p_i $. The influx (and hence the production) then declines exponentially,
  with time constant $ W_{ei} / (J p_i) $, $ J $ being the aquifer's total
  productivity. This is closed-loop control, implemented as a
  `minires.ResSim.well_controls` override, as in `examples.well_control`.

In the figure: the oil saturation under the constant-pressure aquifer as the
front advances (left), and under the Fetkovich aquifer at the end, the sweep
having stalled as it depleted (middle) -- the aquifer's contact stroked along
the boundary faces (`minires.plotting.Plot2D.plt_faces`) in place of its
ring of well markers (hidden by `wells=dict(exclude=...)`); and the influx
rates and aquifer pressure of the two models (right).
"""

from mpl_tools.place import freshfig
import numpy as np
from scipy.ndimage import uniform_filter as smooth

from minires import ResSim
from minires.plotting import show

rng = np.random.default_rng(3)  # Reproducibility (the values are regression tested)

## An elliptic reservoir, incompressible, heterogeneous
p_i, p_bh = 1., 0.  # initial aquifer pressure, and the producer's
dt, nSteps = .02, 120
tt = dt * np.arange(1, nSteps + 1)
logK = 3 * smooth(smooth(rng.standard_normal((40, 40))))

def make(cls=ResSim):
    model = cls(Lx=1, Ly=1, Nx=40, Ny=40, K=np.exp(logK))
    X, Y = model.mesh
    model.active = ((X - .5) / .45)**2 + ((Y - .5) / .3)**2 <= 1
    # The boundary cells: active, with a neighbour that is not (or is off-grid)
    act = np.pad(model.active, 1, constant_values=False)
    boundary = model.active & ~(act[:-2, 1:-1] & act[2:, 1:-1] & act[1:-1, :-2] & act[1:-1, 2:])
    contact = boundary & (X < .25)  # the western end of the ellipse is the aquifer's
    model.wells = [dict(name="Aq", xy=np.column_stack([X[contact], Y[contact]]),
                        aquifer=True, bhp=p_i),
                   dict(name="Prd", xy=[.85, .5], bhp=p_bh, rw=1e-3)]
    return model

## The Fetkovich aquifer: its pressure falls with the cumulative influx
W_ei = .25  # the capacity (about half the pore volume)

class Fetkovich(ResSim):
    def well_controls(self, S, P, k):
        ctrl = super().well_controls(S, P, k)
        aq = self.wells.group == 0                          # the aquifer's completions
        W_e = self.wells.actual_rates[aq, :k].sum() * dt    # cumulative influx so far
        ctrl["bhp"][aq] = p_i * (1 - W_e / W_ei)
        return ctrl

## Simulate
infinite, finite = make(), make(Fetkovich)
S0 = np.zeros(infinite.Nxy)
SS_inf, PP_inf = infinite.sim(dt, nSteps, S0, pbar=False)
SS_fin, PP_fin = finite.sim(dt, nSteps, S0, pbar=False)
q_inf, q_fin = infinite.wells.rates_by_well[0], finite.wells.rates_by_well[0]
p_aq = finite.wells.actual_bhp[0]  # the aquifer's, as controlled

# The aquifer supplies the producer exactly (incompressible), and its water shows up
assert np.allclose(infinite.wells.rates_by_well.sum(0), 0)
assert np.allclose(finite.wells.rates_by_well.sum(0), 0)
assert SS_inf[-1].max() > .5 and (q_inf > 0).all()
# The finite aquifer depletes: exponentially, on the time scale W_ei / (J p_i),
# where J is the aquifer's initial productivity, q_fin[0] / (p_i - p_bh)
assert q_fin[-1] < .2 * q_fin[0] and (np.diff(p_aq) < 0).all()
tau = W_ei / (q_fin[0] / (p_i - p_bh))
assert np.isclose(np.log(q_fin[0] / q_fin[-1]) / (tt[-1] - tt[0]), 1 / tau, rtol=.25)

## Plot
fig, axs = freshfig("Aquifer", ncols=3, figsize=(12, 3.8))
kws: dict = dict(finalize=False, labels=False, colorbar=False)
k = nSteps // 3
kws["wells"] = dict(exclude=["Aq"])  # the contact is drawn as a stroke instead
xy_aq = infinite.wells.xy[infinite.wells.group == 0]
for ax, model, S, title in [(axs[0], infinite, SS_inf[k], f"const. pressure), t = {k*dt:.2f}"),
                            (axs[1], finite, SS_fin[-1], f"Fetkovich), t = {nSteps*dt:.2f}")]:
    model.plt_field(ax, S, "oil", title="Oil sat. (" + title, **kws)
    model.plt_faces(ax, xy_aq, label="Aquifer")
axs[0].legend(loc="lower right", fontsize="small")
ax = axs[2]
ax.plot(tt, q_inf, label="Influx, constant pressure")
ax.plot(tt, q_fin, label="Influx, Fetkovich")
ax.plot(tt, q_fin[0] * np.exp(-(tt - tt[0]) / tau), "k--", lw=1,
        label=r"$q_0 \, e^{-t / \tau}$, $\tau = W_{ei} / (J p_i)$")
ax.set(xlabel="Time", ylabel="Aquifer influx rate", ylim=(0, None))
ax.legend(loc="upper right", fontsize="small")
ax2 = ax.twinx()
ax2.plot(tt, p_aq, "C1:", label="Aquifer pressure (Fetkovich)")
ax2.set(ylabel="Aquifer pressure", ylim=(0, None))
ax2.legend(loc="center right", fontsize="small")
fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(sat_inf   = SS_inf[-1][infinite.active.ravel()],
                  sat_fin   = SS_fin[-1][finite.active.ravel()],
                  q_inf     = q_inf,
                  q_fin     = q_fin,
                  p_aq      = p_aq)

if __name__ == "__main__":
    show()
