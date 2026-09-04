"""The two collages of the README and the docs, composed from the examples' results.

Run from the repo root: `uv run pdoc_template/collage.py` writes `collage.png` and
`collage_features.png` there (they are committed, for the README). `build.py` calls
`make` too, and copies the files into `docs/`, where the package and examples pages
show them.

Both re-plot *results* of the example modules (imported here, so they get run if they
have not been already), rather than pasting their saved figures, so that the panels
share one look: square, uniform fonts, no colorbars.

- `collage.png` (the README's banner): one simulation, `examples.water_cut_gradient`,
  read left to right -- the permeability, the pressure it gives, the water it moves,
  and the adjoint's sensitivity of the production to it.
- `collage_features.png`: one panel per feature, mostly a single axes of some
  example's figure, redrawn.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.layout_engine import ConstrainedLayoutEngine

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))  # makes `examples` importable


def example(name: str) -> ModuleType:
    """Import (i.e. run, unless already run) `examples.<name>`."""
    fignums = plt.get_fignums()
    module = importlib.import_module(f"examples.{name}")
    # Discard the figures the example made (it was imported for its data).
    for num in plt.get_fignums():
        if num not in fignums:
            plt.close(num)
    return module


def field(model, ax, Z, style="default", **kws) -> Any:
    """`plt_field` without colorbar, axis labels, or ticks."""
    kws = dict(colorbar=False, labels=False, finalize=False,
               wells=dict(size=.5, text=False)) | kws
    cc = model.plt_field(ax, Z, style, **kws)
    ax.set(xticks=[], yticks=[])
    return cc


def hero(figsize=(16, 4.4)):
    """One waterflood, from its permeability to the sensitivity of its production."""
    wcg = example("water_cut_gradient")
    model, k, t = wcg.model, wcg.k, wcg.k * wcg.dt
    fig, axs = plt.subplots(ncols=4, figsize=figsize,
                            layout=ConstrainedLayoutEngine(wspace=.06))

    field(model, axs[0], wcg.logK, cmap="viridis", levels=17,
          title="Permeability, $\\log K$")

    # The colour scale is clipped at the wells (the cmap's `over`/`under` make
    # `plt_field` extend rather than blank), whose spikes would otherwise take
    # all the levels, leaving the field between them a single colour.
    P = wcg.PP[k]
    lo, hi = np.percentile(P, [2, 98])
    levels = np.linspace(lo, hi, 17)
    cmap = plt.get_cmap("magma")
    cmap = cmap.with_extremes(over=cmap(1.0), under=cmap(0.0))
    field(model, axs[1], P, cmap=cmap, levels=levels, title="Pressure")
    axs[1].contour(P.reshape(model.shape).T, levels=levels, colors="w",
                   linewidths=.6, alpha=.6, origin="lower",
                   extent=(0, model.Lx, 0, model.Ly))

    field(model, axs[2], wcg.SS[k], "oil", title=f"Oil saturation, t = {t:.2f}")

    m = np.percentile(abs(wcg.G), 98)
    cmap = plt.get_cmap("RdBu_r")
    cmap = cmap.with_extremes(over=cmap(1.0), under=cmap(0.0))
    field(model, axs[3], wcg.G, cmap=cmap, levels=np.linspace(-m, m, 21),
          title="Adjoint: ∂(water cut) / ∂ $\\log K$")

    for ax in axs:
        ax.title.set_fontsize(13)
    return fig


def features(figsize=(16, 12.6)):
    """One panel per feature."""
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=figsize, layout="constrained")
    axs = axs.ravel()
    leg: dict = dict(fontsize="x-small")
    lbl: dict = dict(fontsize="small")

    ## Buckley--Leverett: the verification, and the convergence
    bl = example("buckley_leverett")
    p, tD = bl.profiles["A"], bl.tD_snap["A"]
    ax = axs[0]
    ax.plot(p["xD"], p["exact"], "k-", lw=2, label="Analytic (Buckley–Leverett)")
    ax.plot(p["xD"], p["explicit"], "C0.", ms=4, label="Explicit (upwind)")
    ax.plot(p["xD"], p["implicit"], "C1.", ms=4, label="Implicit (Newton)")
    ax.set(title=f"Verified against Buckley–Leverett ($t_D$ = {tD})",
           xlabel="$x_D$", ylabel="Water saturation")
    ax.legend(**leg)

    ax = axs[1]
    ax.loglog(bl.NN, bl.L1, "C0-o", label="$L_1$ error")
    ax.loglog(bl.NN, np.exp(np.polyval(bl.fit, np.log(bl.NN))), "C0--", lw=1,
              label=f"Fit: $\\propto h^{{{bl.rate:.2f}}}$")
    ax.loglog(bl.NN, bl.L1[0] * bl.NN[0]/bl.NN, "k:", lw=1, label="$O(h)$")
    ax.set(title="Converges under refinement", xlabel="Cells, $N_y$",
           ylabel="Mean error",
           xticks=bl.NN, xticklabels=[str(N) for N in bl.NN])
    ax.minorticks_off()
    ax.legend(**leg)

    ## Quarter five-spot: the two transport schemes, as fronts
    q5 = example("quarter_five_spot")
    ax = axs[2]
    model = q5.model
    kws: dict = dict(levels=[.2, .5, .8], origin="lower",
                     extent=(0, model.Lx, 0, model.Ly))
    field(model, ax, q5.SS_exp[-1], "oil", alpha=.25, wells=False)
    for S, c, name in [(q5.SS_exp[-1], "C0", "Explicit (upwind)"),
                       (q5.SS_imp[-1], "C1", "Implicit (Newton)")]:
        ax.contour(S.reshape(model.shape).T, colors=c, **kws)
        ax.plot([], [], c=c, label=name)
    model.well_scatter(ax, model.wells.xy[:1], +1, text=False, size=.5)
    model.well_scatter(ax, model.wells.xy[1:], -1, text=False, size=.5)
    ax.set(title=f"Explicit and implicit transport (t = {q5.dt*q5.nSteps:.1f})",
           xticks=[], yticks=[], aspect="equal")
    ax.legend(title="Fronts: $s$ = 0.2, 0.5, 0.8", title_fontsize="x-small",
              loc="lower right", **leg)

    ## Rate scheduling
    rs = example("rate_scheduling")
    field(rs.model, axs[3], rs.SS[-1], "oil",
          title=f"Scheduled injection rates (t = {rs.dt*rs.nSteps:.1f})")

    ## Well paths: the sweep, and the allocation along the path
    wp = example("well_path")
    field(wp.path, axs[4], wp.SS_path[-1], "oil", wells=dict(size=.3, text=False),
          title=f"Well paths (t = {wp.dt*wp.nSteps:.1f})")

    ax = axs[5]
    ax.plot(wp.yy, wp.path.wells.actual_rates[wp.inj, -1],
            label="Rate-controlled: $w \\propto WI$")
    for k, ls in [(0, ":"), (wp.nSteps - 1, "-")]:
        ax.plot(wp.yy, wp.onbhp.wells.actual_rates[wp.inj, k], ls, c="C1",
                label=f"BHP-controlled, t = {wp.tt[k]:.2f}")
    ax.set(title="Rate allocation along the path", xlabel="y (along the path)",
           ylabel="Injection rate per completion", ylim=0)
    ax.legend(**leg)

    ## Well control: rate, BHP, and rate with a BHP limit
    wc = example("well_control")
    ax = axs[6]
    ax.plot(wc.tt, wc.prod_rate, label="Rate-controlled")
    ax.plot(wc.tt, wc.prod_bhp, label="BHP-controlled")
    ax.plot(wc.tt, wc.prod_lim, ":", lw=2, label="Rate, with a BHP limit")
    ax.plot(wc.tt, wc.prod_bhp[-1]*np.exp((wc.tt[-1] - wc.tt)/wc.tau), "k--", lw=1,
            label="$\\propto e^{-t/\\tau}$, $\\tau = c_t V_p / J$")
    ax.set(title="Rate, BHP, and limited control", xlabel="Time",
           ylabel="Production rate", yscale="log")
    ax.legend(**leg)

    ## Buildup: a well test, in metric units
    bu = example("buildup")
    ax = axs[7]
    for r in [0, 200, 500, 1000]:
        i = bu.model.xy2ind(bu.L/2 + r, bu.L/2)
        ax.plot(bu.tt, bu.PP[:, i], label=f"r = {r} m")
    ax.plot(bu.tt, bu.p_mean, "k--", lw=1, label="Mean, $\\bar{p}$")
    ax.axvline(bu.kShut*bu.dt, c="k", lw=1, alpha=.4)
    ax.annotate("shut-in", (bu.kShut*bu.dt, bu.PP.min()), xytext=(4, 0),
                textcoords="offset points", **lbl)
    ax.set(title=f"Well test in metric units (K = {bu.perm_est:.0f} mD)",
           xlabel=f"Time{bu.unit_t}", ylabel=f"p{bu.unit_p}")
    ax.legend(**leg)

    ## Compressibility: finite-speed pressure propagation
    pd = example("pressure_diffusion")
    k = 3
    dP = pd.PP[k] - pd.P0
    vmax = abs(dP).max()
    field(pd.model, axs[8], dP, cmap="RdBu_r",
          levels=np.linspace(-vmax, vmax, 21), wells=dict(size=.4, text=False),
          title=f"Compressible: pressure diffusion (t = {k*pd.dt:.4f})")

    ## Compressibility: primary depletion
    dp = example("depletion")
    ax = axs[9]
    ax.plot(dp.tt, dp.p_mean, label="Mean, $\\bar{p}$")
    ax.plot(dp.tt, dp.p_cell, label="Producer cell, $p_\\mathrm{cell}$")
    ax.plot(dp.tt, 1 - dp.q*dp.tt/(dp.model.ct*dp.pore_volume), "k--", lw=1,
            label="Material balance, $p_0 - q t / (c_t V_p)$")
    ax.set(title="Primary depletion (no injection)", xlabel="Time", ylabel="p")
    ax.legend(**leg)

    ## Compressibility: under-injection
    vr = example("voidage_replacement")
    ax = axs[10]
    for SS, vrr, t_bt in zip([vr.SS_full, vr.SS_half], ["1", "½"], vr.breakthrough):
        h, = ax.plot(vr.tt, 1 - SS[:, vr.iprd], label=f"VRR = {vrr}")
        ax.axvline(t_bt, c=h.get_color(), ls=":", lw=1)
    ax.set(title="Under-injection defers breakthrough", xlabel="Time",
           ylabel="Oil saturation in the producer")
    ax.legend(**leg)

    ## Adjoint: history matching
    hm = example("history_match_gradient")
    ax = axs[11]
    for i in range(len(hm.producers)):
        ax.plot(hm.tt, hm.obs[:, i], "*", c=f"C{i}")
        ax.plot(hm.tt, hm.fw_final[:, i], "-", c=f"C{i}")
    ax.plot(hm.tt, hm.fw_prior, "-", c="gray", lw=1, alpha=.7)
    for fmt, label in [("k*", "Observed"), ("k-", "Matched"), ("-", "Prior")]:
        ax.plot([], [], fmt, c="gray" if label == "Prior" else "k", label=label)
    ax.set(title=f"Adjoint history matching ({len(hm.JJ) - 1} descent steps)",
           xlabel="Time", ylabel="Water cut at the 4 producers", ylim=(-.02, 1))
    ax.legend(loc="upper left", **leg)

    for ax in axs:
        ax.set_box_aspect(1)
        ax.title.set_fontsize(11)
        ax.xaxis.label.set_fontsize("small")
        ax.yaxis.label.set_fontsize("small")
        ax.tick_params(labelsize="small")
    return fig


def make(out: Path = root, dpi: int = 100) -> list[Path]:
    """Write both collages into `out`; return their paths."""
    files = []
    for name, maker in [("collage", hero), ("collage_features", features)]:
        fig = maker()
        files.append(out / f"{name}.png")
        fig.savefig(files[-1], dpi=dpi)
        plt.close(fig)
    return files


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    for file in make():
        print("Wrote", file.relative_to(root))
