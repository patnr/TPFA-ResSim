"""The Egg model, flattened to 2D, against its published 3D solution. **In metric units.**

The [Egg model](https://doi.org/10.1002/gdj3.21) (Jansen et al., 2014) is a synthetic
benchmark reservoir, much used in history-matching and optimization studies: an
egg-shaped region of 60 × 60 × 7 cells (8 × 8 × 4 m, 18553 of them active) of
**channelized** permeability (50-7000 mD; an ensemble of 101 realizations, of which
this is the first) and porosity 0.2, flooded for 3600 days by 8 water injectors at
79.5 m³/day into 4 producers held at 395 bar, from an initial 400 bar. Oil 5 cP,
water 1 cP, both at 1e-5/bar; Corey relative permeabilities with exponents 3 and 4.
The data (`egg.npz`, ref the "Data" section below) are JutulDarcy's copy of the
ECLIPSE deck, which comes with the deck's solution by ECLIPSE 100 -- in 3D, with
gravity -- as its reference.

## The 2D reduction

This simulator is areal, so the 7 layers are **vertically averaged**: for flow along
the layers, the exact upscaling of a stack is the arithmetic mean of their
permeabilities (each carries flux in proportion to its own), the inactive ones
counting as 0 over the 28 m column, and the porosity likewise; the egg's outline
becomes the `ResSim.active` mask. What is lost is
gravity (a mere 0.27 bar of hydrostatic contrast over the column, against a 5 bar
drawdown) and the layers' *individual* channel patterns, through the fastest of
which the water arrives first in 3D. The wells, being vertical through all 7
layers, are just one completion each here; the rate, per unit thickness (ref
`ResSim.cdarcy`), is the 79.5 m³/day over the 28 m.

The deck's own quirks are kept: the water starts at $S_w = 0.1$, *below* the 0.2 at
which it becomes mobile (so the first tenth of a pore volume injected only fills up
the immobile saturation), and the injectors' 420 bar limit, which never binds. The
relative permeabilities are the deck's Corey curves (`minires.fluids.Fluid`): exponents 3
and 4 with end-points 0.6 and 0.8 at the residuals $ S_{wc} = 0.2 $, $ S_{or} = 0.15 $
(the table runs on to $ S_w = 0.9 $, but nothing but water moves there). Their
fractional flow is steeper than the quadratic default's (its maximal slope 5.65,
against 3.5), which the CFL estimate follows (`ResSim.estimate_1CFL`).

## Validation

The 2D model reproduces the 3D reference to within an RMS of **0.01** in the water
cut of every producer, and to within 4-6% on average in their oil rates; the
recovery after 3600 days is 0.57 of the oil in place, against 0.59. Breakthrough
comes 15-30 days later than in 3D at three of the four producers -- the vertical
averaging smoothing the fastest layer's channel away, as said -- and PROD2, the
strongest well, produces 7% less oil.

That a 2D areal model gets this close is because the reservoir is thin (28 m
against 480 m across), the wells penetrate all of it, and the channels are largely
stacked, so that the flow is nearly layer-parallel. It is *not* generic: had the
layers had separate channel systems, the average would have connected wells that
none of the layers connects.

The same flattened model run in JutulDarcy (a fully implicit code, reading the
reduction as an ECLIPSE deck) agrees with this one to within 0.007 RMS in the water
cut, and lies as far from the 3D original as this one does (0.008 against 0.009):
the residual is the flattening, not the simulator.

In the figures:

- "fields" (left): the vertically averaged permeability, its channels
  (the yellow bands) running roughly N-S; injectors are black triangles, producers
  white. Outside the egg the cells are inactive (`ResSim.active`), hence blank.
- "fields" (right three): the oil being displaced -- fastest along the channels,
  and first towards PROD2 (the producer sitting on the widest one, near the centre),
  which breaks through after 6 months.
- "production" (left): the water cut, this model (solid) against the 3D reference
  (dashed), per producer. The curves lie on top of each other at the scale of the
  plot; the 2D breakthroughs come slightly later.
- "production" (right): the oil rate, on a log axis, likewise; the 3D reference's
  early points are its finer initial report steps. PROD2's initial rate is set by
  its well index -- the same Peaceman formula in both codes -- and its later
  decline by how fast the water arrives.

## Data

`egg.npz` (73 KB) holds, `x`-first as the simulator orders its fields: `perm`, the
deck's `PERMX` in mD, `(60, 60, 7)` = `(Nx, Ny, Nz)`, `float32` (`PERMY = PERMX` and
`PERMZ = 0.1 PERMX` in the deck, irrelevant in 2D); `active`, its `ACTNUM`, `bool`;
and the reference solution's 135 report times (`ref_time`, in days) with its oil and
water production rates (`ref_orat`, `ref_wrat`, `(135, 4)`, m³/day, `PROD1..4`).
Everything else the model needs -- cell size, porosity, fluids, wells, controls -- is
a handful of numbers, kept in this script.

The source is the deck as JutulDarcy ships it (`GeoEnergyIO`'s `EGG` test input:
`EGG.DATA` with `MDARCY.INC`, `ACTIVE.INC`, and `REFERENCE.CSV`, the ECLIPSE 100
solution in SI, production signed negative), i.e. realization 1 of the 101 published
by TU Delft (doi:10.4121/uuid:916c86cd-3558-4672-829a-105c62985ab2, 4TU General
Terms of Use). With `d` that directory, the file was made by

    import re, numpy as np
    def keyword(path, kw):                  # a GRID keyword's numbers, as (Nx, Ny, Nz)
        body = re.sub(r"--.*", "", open(path).read().split(kw, 1)[1].split("/", 1)[0])
        vals = [float(v) for tok in body.split() for n, _, v in [tok.rpartition("*")]
                for _ in range(int(n) if n else 1)]              # ECLIPSE's `n*value`
        return np.array(vals).reshape(7, 60, 60).transpose(2, 1, 0)   # I fastest
    perm = keyword(f"{d}/MDARCY.INC", "PERMX")
    active = keyword(f"{d}/ACTIVE.INC", "ACTNUM").astype(bool)
    ref = np.genfromtxt(f"{d}/REFERENCE.CSV", delimiter=",", names=True)
    prods = [f"PROD{i}" for i in range(1, 5)]
    orat = -np.column_stack([ref[w + "orat"] for w in prods]) * 86400
    wrat = -np.column_stack([ref[w + "wrat"] for w in prods]) * 86400
    wrat[np.abs(wrat) < 1e-9] = 0                               # the reference's round-off
    np.savez_compressed("egg.npz", perm=perm.astype(np.float32), active=active,
                        ref_time=ref["time"] / 86400, ref_orat=orat, ref_wrat=wrat)
"""

from pathlib import Path

from mpl_tools.place import freshfig
import numpy as np

from minires import ResSim
from minires.plotting import show

## The data: realization 1 of the Egg ensemble (ref the module docstring)
data = np.load(Path(__file__).with_name("egg.npz"))
perm, active = data["perm"], data["active"]       # (Nx, Ny, Nz) = (60, 60, 7)
Nx, Ny, Nz = perm.shape
h, dz = 8.0, 4.0                                  # cell size [m]
H = Nz*dz                                         # thickness [m]

## The 2D reduction: vertical averaging
# For flow along the layers, the exact upscaling of a stack of layers is the
# arithmetic mean of their permeabilities (each carries flux in proportion to
# its own), with the inactive layers (`K = 0`) counted in the thickness. The pore
# volume is likewise the column's. The egg-shaped footprint -- the cells active in
# any layer -- becomes the model's `active` mask; outside it, `K` and `por` are
# immaterial (but `K` must be finite and positive, so the mean is filled in).
K   = (perm * active).sum(-1) / Nz
por = 0.2 * active.mean(-1)
footprint = active.any(-1)
K   = np.where(footprint, K, K[footprint].mean())

## Wells: 8 injectors on rate, 4 producers on BHP, from the deck's `WELSPECS`
# (1-based cell indices) and `SCHEDULE`. The rate is per unit thickness (ref
# `ResSim.cdarcy`), the whole column being one cell here.
ij = dict(INJECT1=(5, 57), INJECT2=(30, 53), INJECT3=(2, 35), INJECT4=(27, 29),
          INJECT5=(50, 35), INJECT6=(8, 9), INJECT7=(32, 2), INJECT8=(57, 6),
          PROD1=(16, 43), PROD2=(35, 40), PROD3=(23, 16), PROD4=(43, 18))
q_inj, p_prod, p_max, rw = 79.5/H, 395, 420, 0.2   # m²/day, bar, bar, m
wells = {name: dict(xy=[(i - .5)*h, (j - .5)*h], rw=rw,
                    **(dict(rate=q_inj) if name.startswith("INJ") else dict(bhp=p_prod)))
         for name, (i, j) in ij.items()}
producers = [i for i, name in enumerate(ij) if name.startswith("PROD")]

## Fluids: 1 / 5 cP, both 1e-5/bar (rock incompressible), and the deck's `SWOF`
C = 86400 * 9.869233e-16 * 1e5 / 1e-3               # m, day, bar, mD, cP: 0.008527
Sw0 = 0.1                                           # initial water saturation
p0 = 400                                            # initial pressure [bar]


# The deck's `SWOF` table is Corey (ref `minires.fluids.Fluid`): exponents 3 (water) and 4
# (oil), end-points 0.6 and 0.8, water immobile below Sw = 0.2 and oil below So = 0.15.
fluid = dict(vw=1, vo=5, swc=0.2, sor=0.15, nw=3, no=4, krw0=0.6, kro0=0.8)

model = ResSim(Lx=Nx*h, Ly=Ny*h, Nx=Nx, Ny=Ny, cdarcy=C, K=K, por=por, active=footprint,
               ct=1e-5, fluid=fluid, wells=wells)

## Simulate 3600 days in 30-day steps
dt, nSteps = 30, 120
tt = dt * np.arange(1, nSteps + 1)
SS, PP = model.sim(dt, nSteps, np.full(model.Nxy, Sw0), P0=np.full(model.Nxy, p0), pbar=False)

# The injectors' 420 bar limit (which the deck also has) never binds, so rate
# control is the whole story, as in the reference.
bhp_inj = model.wells.actual_bhp[:8]
assert bhp_inj.max() < p_max, "the injectors' BHP limit binds"

## Production, per well: water cut and oil rate
cells = model.xy2ind(*model.wells.xy[producers].T)
cut = model.fluid.fractional_flow(SS[1:][:, cells])   # (nSteps, 4)
rate = -model.wells.actual_rates[producers].T * H        # m³/day, total, positive
oil = rate * (1 - cut)

## The reference: the 3D model (ECLIPSE 100, via JutulDarcy's copy of the deck)
ref_t, ref_o, ref_w = data["ref_time"], data["ref_orat"], data["ref_wrat"]
ref_cut = ref_w / (ref_w + ref_o)
sel = ref_t >= dt                                        # our first report time
cut_at_ref = np.column_stack([np.interp(ref_t[sel], tt, cut[:, j]) for j in range(4)])
oil_at_ref = np.column_stack([np.interp(ref_t[sel], tt, oil[:, j]) for j in range(4)])
rms_cut = np.sqrt(np.mean((cut_at_ref - ref_cut[sel])**2, axis=0))
rel_oil = np.abs(oil_at_ref - ref_o[sel]).mean(axis=0) / ref_o[sel].mean(axis=0)
ooip = 0.2 * h*h*dz * active.sum() * (1 - Sw0)                      # m³
recovery = oil.sum()*dt / ooip
recovery_ref = np.trapezoid(ref_o, ref_t, axis=0).sum() / ooip
assert (rms_cut < 0.02).all() and (rel_oil < 0.1).all(), "the 2D model has drifted from the 3D reference"

print(f"RMS water-cut misfit per producer: {rms_cut.round(3)}")
print(f"Mean relative oil-rate misfit per producer: {rel_oil.round(3)}")
print(f"Recovery (fraction of OOIP): {recovery:.3f}, 3D reference {recovery_ref:.3f}")
print(f"Injector BHP: {bhp_inj.min():.1f}-{bhp_inj.max():.1f} bar")

## Plot: the permeability, and the water's advance
fig, axs = freshfig("Egg -- fields", ncols=4, figsize=(15, 4), sharex=True, sharey=True)
kws: dict = dict(colorbar=False, finalize=False, wells=dict(size=.4, text=False))
model.plt_field(axs[0], np.log10(np.where(footprint, K, np.nan)).ravel(), cmap="viridis",
                levels=np.linspace(2, 3.7, 18), title="$\\log_{10} K$ [mD]", **kws)
for ax, k in zip(axs[1:], [20, 60, nSteps]):
    model.plt_field(ax, SS[k], "oil", title=f"Oil saturation, t = {k*dt} days", labels=False, **kws)
fig.tight_layout()

## Plot: production, against the 3D reference
fig, (ax1, ax2) = freshfig("Egg -- production", ncols=2, figsize=(12, 4))
names = [list(ij)[i] for i in producers]
for j, name in enumerate(names):
    ax1.plot(tt, cut[:, j], c=f"C{j}", label=name)
    ax1.plot(ref_t, ref_cut[:, j], "--", c=f"C{j}", lw=1)
    ax2.plot(tt, oil[:, j], c=f"C{j}", label=name)
    ax2.plot(ref_t, ref_o[:, j], "--", c=f"C{j}", lw=1)
ax1.plot([], [], "k-", label="2D (this)")
ax1.plot([], [], "k--", lw=1, label="3D (ECLIPSE 100)")
ax1.set(title="Water cut", xlabel="Time [day]", ylabel="$f_w$")
ax1.legend(fontsize="small", ncols=2)
ax2.set(title="Oil rate", xlabel="Time [day]", ylabel="[m³/day]", yscale="log")
fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(water_cut = cut,
                  oil_rate  = oil,
                  S_final   = SS[-1],
                  bhp_inj   = bhp_inj[:, -1],
                  rms_cut   = rms_cut,
                  rel_oil   = rel_oil,
                  recovery  = [recovery, recovery_ref])

if __name__ == "__main__":
    show()
