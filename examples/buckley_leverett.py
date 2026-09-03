"""The Buckley--Leverett solution -- the one *exact* answer we can be checked against.

Every other example here is an *illustration*: we plot what the code does, and
argue that it looks right. This one is a **verification**: in 1D, with constant
total velocity, the water equation has an analytic solution
(Buckley & Leverett, 1942), so the numerical profile can be compared with the
truth, and the error made to shrink under refinement.

**The construction.** With $ ∇ ⋅ v = 0 $ the transport equation collapses to the
scalar conservation law
$$ φ \\, ∂s/∂t + v \\, ∂f(s)/∂x = 0 \\,, $$
whose characteristics carry a given $s$ at speed $ v f'(s)/φ $. Since $f$ is
S-shaped, $f'$ is *not* monotone, so those characteristics cross: the profile
would become multivalued, and a **shock** forms instead. Its saturation,
$S_f$, is fixed by requiring that the shock speed (from mass balance across it)
equal the characteristic speed just behind it -- which is the **Welge (1952)
tangent** construction: the chord from the initial state to $(S_f, f(S_f))$ must
be tangent to $f$,
$$ \\frac{f(S_f)}{S_f - s_\\mathrm{wc}} = f'(S_f) \\,. $$
Everything else follows: the shock travels at that chord's slope, breakthrough
occurs when it reaches the outlet, and thereafter the outlet saturation is read
off $ f'(s) = 1/t_D $ (which is Welge's production forecast).

**Closed form for this model.** `ResSim.RelPerm` being quadratic, the tangent
condition can be solved by hand. In terms of the normalized saturation
(ref `ResSim.rescale_sat`) and the endpoint mobility ratio $ M = v_o/v_w $,
$$ S_f^* = \\frac{1}{\\sqrt{1 + M}} \\,, \\qquad
   t_D^\\mathrm{bt} = (1 - s_\\mathrm{wc} - s_\\mathrm{or})
       \\, \\frac{2 \\, (1 + M - \\sqrt{1+M})}{M \\, \\sqrt{1+M}} \\,, $$
so for the default unit-viscosity fluids $ S_f = 1/\\sqrt{2} ≈ 0.7071 $ and
breakthrough comes at $ 2(\\sqrt{2}-1) ≈ 0.8284 $ pore volumes injected. Below,
the tangent is located *numerically* (from the model's own `RelPerm` and
`dRelPerm`) and asserted to agree with these -- so the check cuts both ways:
it validates the analytic solution we then compare the simulation against.

Notes on the setup:

- Time is measured in **pore volumes injected** ($t_D$), and distance in
  fractions of the length ($x_D$). The rate, pore volume and length are all
  $1$ here, so $ t_D = t $ and $ x_D = y $ -- no scaling clutters the plots.
- The 1D domain is a single **column** of cells (`Nx=1`), i.e. the flow is
  along $y$. The transpose (`Ny=1`) is not available: `TPFA` places its
  off-diagonals at offsets $ ±N_y $ and $ ±1 $, which collide when $ N_y = 1 $.
- The ends are **wells** (a source and a sink), not boundary conditions, which
  costs two $O(h)$ discrepancies with the textbook problem: the inlet cell only
  approaches $ s = 1 - s_\\mathrm{or} $ (it is filled at a finite rate, not
  held at a value), and the outlet is sampled half a cell short of $ x_D = 1 $.
  Both are visible below, and both vanish under refinement.

In the figures:

- "fractional flow": the construction itself. The tangent from the initial
  state finds $ S_f = 0.71 $ for the unit-viscosity fluids, but only $0.44$ for
  the case with $ M = 5 $ -- a *more adverse* mobility ratio gives a *weaker*
  shock, arriving sooner (at $ t_D = 0.35 $ rather than $0.83$), which is the
  whole reason mobility ratio matters to a waterflood. Where the tangent
  reaches $ f = 1 $ (open squares) is Welge's second reading of it: the
  *average* saturation behind the front -- and hence, nothing having been
  produced yet, the recovery at breakthrough. Asserted against the simulation
  below.
- "saturation profile": the numerical and analytic profiles, at a time before
  breakthrough. The explicit scheme follows the rarefaction to within 0.017 in
  saturation (0.005 on average), and smears the shock over some 4 cells. The
  implicit scheme is four times as far off in the rarefaction, and spreads the
  shock over five times as many cells -- as in `quarter_five_spot.py`, it is
  the more diffusive of the two. Note that neither *overshoots*.
- "verification" (left): the water cut at the producer, against Welge's
  forecast. It breaks through a shade early (the producer being half a cell
  inside), then follows the analytic curve to within 0.004.
- "verification" (right): the $L_1$ error of the profile, which shrinks as
  $ h^{0.87} $ -- first-order convergence, all but the last 13% of it, the
  shortfall being the usual one for a monotone first-order scheme resolving a
  discontinuity. (The zig-zag about that fit is also expected: the error
  depends on where the shock happens to fall within a cell.) The rate is what
  matters: it certifies that the scheme is consistent, i.e. that the error is
  *discretization*, not *bug*.
"""

from mpl_tools.place import freshfig
import numpy as np
from scipy.optimize import minimize_scalar

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show

## Setup


def make_model(N: int, fluid: dict) -> ResSim:
    """A 1D column of `N` cells: injector at the bottom, producer at the top.

    Unit length, unit pore volume, unit rate -- so that time *is* $t_D$
    (pore volumes injected), and position *is* $x_D$.
    """
    return ResSim(Lx=1, Ly=1, Nx=1, Ny=N, **fluid,
                  wells=[dict(xy=[0, 0], rate=+1),
                         dict(xy=[0, 1], rate=-1)])


## The analytic solution
# NB: built from the model's *own* `RelPerm`/`dRelPerm`, so that it cannot
# drift from the simulator's notion of the fluids -- only from its numerics.


def frac_flow(model, s):
    """Fractional flow of water, $ f = λ_w / (λ_w + λ_o) $."""
    Mw, Mo = model.RelPerm(s)
    return Mw / (Mw + Mo)


def d_frac_flow(model, s):
    """Derivative $ f'(s) $, i.e. the speed of the characteristic carrying `s`."""
    Mw, Mo = model.RelPerm(s)
    dMw, dMo = model.dRelPerm(s)
    return (dMw*Mo - Mw*dMo) / (Mw + Mo)**2


def welge_tangent(model) -> tuple:
    """Locate the shock: `(S_f, shock speed)`, ref the docstring.

    Rather than solving $ f(S)/(S - s_\\mathrm{wc}) = f'(S) $, we *maximize*
    the chord slope -- the same point, but needing no derivative, and with no
    root-bracketing to get wrong.
    """
    lo, hi = model.swc, 1 - model.sor
    chord = lambda S: frac_flow(model, S) / (S - lo)  # noqa: E731
    opt = minimize_scalar(lambda S: -chord(S), method="bounded",
                          bounds=(lo + 1e-9, hi), options=dict(xatol=1e-12))
    return opt.x, chord(opt.x)


def analytic(model, tD, xD, S_f):
    """The saturation at `xD` at time `tD`: the profile, and its shock.

    Behind the shock, $ x_D = t_D \\, f'(s) $ -- monotone in $s$ over
    $ [S_f, 1 - s_\\mathrm{or}] $ (the tangent point lying beyond $f$'s
    inflection), so it is inverted by interpolation. The shock itself needs no
    special treatment: the tangent condition puts its position at exactly the
    end of that range, so anything ahead of it is simply `right=swc`.
    """
    ss = np.linspace(S_f, 1 - model.sor, 10001)
    xx = tD * d_frac_flow(model, ss)
    return np.interp(xD, xx[::-1], ss[::-1], right=model.swc)


## Verify the analytic solution against the closed form
cases: dict = dict(
    A=dict(),                              # defaults: vw = vo = 1, swc = sor = 0
    B=dict(vo=5., swc=.2, sor=.2),         # a contrast in both viscosity and endpoints
)

models = {case: make_model(200, fluid) for case, fluid in cases.items()}

for case, model in models.items():
    S_f, speed = welge_tangent(model)
    # The closed form (ref the docstring), in terms of `M` and the endpoints
    M = model.vo / model.vw
    span = 1 - model.swc - model.sor
    S_f_exact = model.swc + span / np.sqrt(1 + M)
    tD_bt_exact = span * 2*(1 + M - np.sqrt(1 + M)) / (M * np.sqrt(1 + M))
    assert np.isclose(S_f, S_f_exact, rtol=1e-8), "Welge tangent misplaced."
    assert np.isclose(1/speed, tD_bt_exact, rtol=1e-8), "Closed form disagrees."
    print(f"Case {case}: M = {M:g}, S_f = {S_f:.4f}, breakthrough at {1/speed:.4f} PVI")

## Simulate: the profile, both schemes, both cases
tD_snap = dict(A=.5, B=.3)  # a time before breakthrough, for each case
profiles: dict = {}

for case, model in models.items():
    S_f, speed = welge_tangent(model)
    xD = model.mesh[1].ravel()
    S0 = np.full(model.Nxy, model.swc)

    nSteps = 50
    dt = tD_snap[case] / nSteps
    S_exp, _ = model.sim(dt, nSteps, S0, pbar=False)
    S_imp, _ = model.sim(dt, nSteps, S0, pbar=False, implicit=True)

    profiles[case] = dict(xD=xD, S_f=S_f, speed=speed,
                          exact=analytic(model, tD_snap[case], xD, S_f),
                          explicit=S_exp[-1], implicit=S_imp[-1])

    # Neither scheme may overshoot the physical range: the analytic solution is
    # bounded by its data, and a monotone scheme must be too.
    for scheme, S in [("explicit", S_exp), ("implicit", S_imp)]:
        assert model.swc - 1e-12 <= S.min() and S.max() <= 1 - model.sor + 1e-12, (
            f"Case {case}, {scheme} scheme: saturation out of bounds.")

## Simulate: the production history (case A, past breakthrough)
model = models["A"]
S_f, speed = welge_tangent(model)
tD_bt = 1 / speed

nSteps = 150
dt = 1.5 / nSteps
tt = dt * np.arange(nSteps + 1)
SS, _ = model.sim(dt, nSteps, np.full(model.Nxy, model.swc), pbar=False)

# The water cut is the fractional flow of the producer's cell -- ref
# `ResSim.assemble_wells`, which is what draws the produced fluid at that ratio.
i_prd = model.xy2ind(*model.wells.xy[1])
water_cut = frac_flow(model, SS[:, i_prd])
# Welge's forecast: the outlet saturation is the one whose characteristic has
# just arrived. `analytic` returns `swc` (whence a zero water cut) before that.
water_cut_exact = np.array([frac_flow(model, analytic(model, t, 1., S_f))
                            for t in tt])
water_cut_exact[0] = 0  # `tD = 0` puts the whole profile at the inlet

# Mass balance: what was injected is either still in place, or was produced.
# (The tolerance is set by the trapezoidal integration of the jump at
# breakthrough, not by the scheme, which conserves mass exactly.)
in_place = (SS[-1] - model.swc).mean()
produced = np.trapezoid(water_cut, tt)
assert np.isclose(in_place + produced, tt[-1], rtol=2e-3), "Water unaccounted for."

# Before breakthrough that balance is *exact*, nothing having been produced.
# NB: "nothing" is not quite `0`: the explicit scheme's stencil advances one
# cell per sub-step, so a (multiplicatively vanishing) tail of the front runs
# ahead of it -- reaching the producer at some $ 10^{-150} $, long before the
# water does. Hence the threshold, which also defines breakthrough below.
DRY = 1e-4  # water cut counting as "no water", i.e. $ s ⪅ 0.01 $
k_pre = round(.7 / dt)
assert water_cut[k_pre] < DRY, "Breakthrough far too early."
assert np.isclose((SS[k_pre] - model.swc).mean(), tt[k_pre]), "Injected water lost."

# Breakthrough should be *slightly* early, the producer sitting half a cell
# short of the outlet -- i.e. by `(hy/2) / speed`, which is under one `dt` here.
tD_bt_sim = tt[(water_cut > DRY).argmax()]
assert 0 <= tD_bt - tD_bt_sim < 2*dt, "Breakthrough mistimed."

# Evaluated at breakthrough, it is the *other* reading of the tangent (ref the
# "fractional flow" figure): the mean saturation is then the average behind the
# front, $ s_wc + t_D^bt $ -- to within the smearing.
k_bt = round(tD_bt / dt)
assert np.isclose(SS[k_bt].mean(), model.swc + tD_bt, rtol=1e-2), (
    "Welge average is off.")

## Simulate: convergence under refinement (case A, explicit scheme)
NN = np.array([50, 100, 200, 400, 800])
L1 = np.zeros(len(NN))

for i, N in enumerate(NN):
    m = make_model(N, cases["A"])
    S, _ = m.sim(tD_snap["A"]/50, 50, np.full(m.Nxy, m.swc), pbar=False)
    # NB: the analytic solution is grid-independent -- only sampled anew
    exact = analytic(m, tD_snap["A"], m.mesh[1].ravel(), profiles["A"]["S_f"])
    L1[i] = abs(S[-1] - exact).mean()

fit = np.polyfit(np.log(NN), np.log(L1), 1)
rate = -fit[0]
print(f"Convergence: L1 error ~ h^{rate:.2f}")
assert .7 < rate < 1.1, "Lost (near-)first-order convergence."

## Plot: the fractional-flow curve and the Welge tangent
fig, ax = freshfig("Buckley-Leverett -- fractional flow", figsize=(6, 5))

for case, p in profiles.items():
    m = models[case]
    S_f, tD_bt_ = p["S_f"], 1 / p["speed"]
    ss = np.linspace(m.swc, 1 - m.sor, 201)
    (h,) = ax.plot(ss, frac_flow(m, ss), lw=2,
                   label=f"$M$ = {m.vo/m.vw:g}, "
                         f"$s_\\mathrm{{wc}}$ = {m.swc:g}, "
                         f"$s_\\mathrm{{or}}$ = {m.sor:g}")
    # The tangent, from the initial state up to `f = 1`, which it reaches at
    # the *average* saturation behind the front (Welge's other reading of it).
    ax.plot([m.swc, m.swc + tD_bt_], [0, 1], ":", c=h.get_color(), lw=1)
    ax.plot(S_f, frac_flow(m, S_f), "o", c=h.get_color(), ms=8,
            label=f"$S_f$ = {S_f:.3f},  $t_D^\\mathrm{{bt}}$ = {tD_bt_:.3f}")
    ax.plot(m.swc + tD_bt_, 1, "s", c=h.get_color(), ms=6, mfc="none",
            label=f"$\\bar{{s}}$ = {m.swc + tD_bt_:.3f} (at breakthrough)")

ax.set(title="The Welge tangent construction", xlabel="Water saturation, $s$",
       ylabel="Fractional flow, $f(s)$", xlim=(0, 1), ylim=(0, 1.08))
ax.legend(fontsize="small", loc="lower right")
fig.tight_layout()

## Plot: the saturation profiles, numerical vs. analytic
fig, axs = freshfig("Buckley-Leverett -- saturation profile", ncols=2,
                    sharey=True, figsize=(10, 4.5))

for ax, (case, p) in zip(axs, profiles.items()):
    ax.plot(p["xD"], p["exact"], "k-", lw=2, label="Analytic (Buckley-Leverett)")
    ax.plot(p["xD"], p["explicit"], "C0.", ms=4, label="Explicit (upwind)")
    ax.plot(p["xD"], p["implicit"], "C1.", ms=4, label="Implicit (Newton)")
    ax.axvline(tD_snap[case] * p["speed"], c="k", ls=":", lw=1,
               label="Shock position, $t_D \\, f'(S_f)$")
    m = models[case]
    ax.set(title=f"Case {case}:  $M$ = {m.vo/m.vw:g},"
                 f"  $t_D$ = {tD_snap[case]}", xlabel="$x_D$")
axs[0].set_ylabel("Water saturation, $s$")
axs[0].legend(fontsize="small")
fig.tight_layout()

## Plot: water cut, and the convergence of the profile
fig, (ax1, ax2) = freshfig("Buckley-Leverett -- verification", ncols=2,
                           figsize=(10, 4.5))

ax1.plot(tt, water_cut_exact, "k-", lw=2, label="Welge forecast")
ax1.plot(tt, water_cut, "C0.", ms=4, label="Simulated (explicit)")
ax1.axvline(tD_bt, c="k", ls=":", lw=1,
            label=f"Breakthrough, $2(\\sqrt{{2}}-1)$ = {tD_bt:.4f}")
ax1.set(title="Water cut at the producer", xlabel="$t_D$ (pore volumes injected)",
        ylabel="$f_w$", ylim=(-.03, 1))
ax1.legend(fontsize="small", loc="lower right")

ax2.loglog(NN, L1, "C0-o", label="$L_1$ error")
ax2.loglog(NN, np.exp(np.polyval(fit, np.log(NN))), "C0--", lw=1,
           label=f"Fit: $\\propto h^{{{rate:.2f}}}$")
ax2.loglog(NN, L1[0] * NN[0]/NN, "k:", lw=1, label="$O(h)$, for reference")
ax2.set(title=f"Convergence of the profile at $t_D$ = {tD_snap['A']}",
        xlabel="$N_y$", ylabel="Mean $|s - s_\\mathrm{exact}|$",
        xticks=NN, xticklabels=[str(N) for N in NN])
ax2.minorticks_off()
ax2.legend(fontsize="small")
fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(explicit  = profiles["A"]["explicit"],
                  implicit  = profiles["A"]["implicit"],
                  case_B    = profiles["B"]["explicit"],
                  welge     = [profiles[c][k] for c in "AB"
                               for k in ["S_f", "speed"]],
                  water_cut = water_cut[water_cut > DRY],
                  bt        = [tD_bt_sim, tD_bt],
                  L1        = L1)

if __name__ == "__main__":
    show()
