"""The gradient of a production-history misfit wrt. $\\log K$, and a few descent steps.

The history-matching case, in its simplest form. A *truth* permeability field
(smoothed, log-normal) produces the *observations*: the water cut at each of
the four producers of a five-spot, at every time step. The *prior* guess is
homogeneous, $\\log K = 0$. The objective is the mean squared error of the
prior's production history against the observations, and
`TPFA_ResSim.tlm.adjoint` gives its gradient with respect to every cell's
$\\log K$ -- checked against a finite difference in a random direction -- for
about the cost of one more simulation, however many cells there are. That is
what makes gradient-based history matching feasible at all, and a few steps of
steepest descent (each with a coarse line search) are taken to show it works:
the misfit falls, and the update goes the way of the truth.

The seeds are those of a data misfit
(ref the "Seeding" section of `TPFA_ResSim.tlm`): for each observed time and
producer, the residual, weighted by the derivative of the observation operator
-- here $ f_w'(s) $ in the producer's cell, via `tlm.fractional_flow`.

In the figure:

- Top left: the truth $\\log K$, which the observations come from.
- Top middle: the gradient of the misfit at the prior. Since the prior is
  homogeneous, this is the whole of the first update (with a minus sign): it
  is negative (raise $K$) where the truth is permeable *and* the water reaches
  a producer through it, positive where the truth is tight -- but only where
  the production data are sensitive, i.e. along the flow paths. Hence its
  correlation with the truth over the whole field is weak (about 0.2; it is
  recorded in the digest): the data are blind to most of the field.
- Top right: the estimate after the descent steps (on its own colour scale:
  the update is far smaller than the truth's variations). It moves towards
  the truth along the flow paths and not elsewhere, the data saying nothing
  about the rest: the classic ill-posedness that a prior (regularization)
  would address.
- Bottom: the production histories -- observations, prior, and the final
  estimate -- and the misfit per iteration.

.. note:: Steepest descent is used for its simplicity, not its merit.

    A Gauss--Newton or quasi-Newton method, and a prior term, would do far
    better; the point here is the *gradient*, which any of them needs.
"""

from mpl_tools.place import freshfig
import numpy as np
from scipy.ndimage import uniform_filter as smooth

from TPFA_ResSim import ResSim
from TPFA_ResSim.plotting import show
from TPFA_ResSim.tlm import adjoint, fractional_flow

rng = np.random.default_rng(1)  # Reproducibility (the values are regression tested)

## Model: a five-spot
wells = [
    dict(xy=[.5, .5], rate=+1  , name="inj"),
    dict(xy=[1 , 1 ], rate=-.25, name="NE"),
    dict(xy=[0 , 1 ], rate=-.25, name="NW"),
    dict(xy=[0 , 0 ], rate=-.25, name="SW"),
    dict(xy=[1 , 0 ], rate=-.25, name="SE"),
]
grid: dict = dict(Lx=1, Ly=1, Nx=32, Ny=32)
dt, nSteps = .05, 30


def new_model(logK):
    model = ResSim(**grid, wells=wells)
    model.K = np.exp(logK)  # isotropic: broadcast to both components
    return model


model = new_model(0)  # the prior: homogeneous
S0 = np.zeros(model.Nxy)
producers = model.wells.names[1:]
prd = model.xy2ind(*model.wells.xy[1:].T)  # their cells


def water_cut(model, SS):
    """`(nSteps, nPrd)` water cut at each producer, at times `1..nSteps`."""
    return np.array([fractional_flow(model, S)[0][prd] for S in SS[1:]])


## The truth, and the observations it produces
logK_true = 3 * smooth(smooth(rng.standard_normal(model.shape)))
truth = new_model(logK_true)
obs = water_cut(truth, truth.sim(dt, nSteps, S0, pbar=False)[0])


## The objective, and its gradient by the adjoint
def misfit(logK, gradient=False):
    """Mean squared error of the production history of `logK` against `obs`."""
    model = new_model(logK)
    SS, PP = model.sim(dt, nSteps, S0, pbar=False)
    fw = water_cut(model, SS)
    residual = fw - obs
    J = (residual**2).mean()
    if not gradient:
        return J
    # Seed: ∂J/∂s_k[i] = 2/nObs * residual * f_w'(s), at the producers, each time
    dJ_dSS = np.zeros_like(SS)
    for k in range(1, nSteps + 1):
        dJ_dSS[k, prd] = 2 / obs.size * residual[k - 1] * fractional_flow(model, SS[k])[1][prd]
    G = adjoint(model, dt, SS, PP, dJ_dSS).logK.sum(0)  # isotropic ⇒ sum the components
    return J, G, fw


logK = np.zeros(model.shape)  # the prior
J0, G0, fw_prior = misfit(logK, gradient=True)

## Check: a finite difference in a random direction of log K
direction = rng.standard_normal(model.shape)
eps = 1e-5
fd = (misfit(logK + eps*direction) - misfit(logK - eps*direction)) / (2*eps)
directional = (G0 * direction).sum()
assert abs(fd - directional) < 1e-4 * abs(directional), (fd, directional)

## A few steps of steepest descent, each with a coarse line search
nIter = 4
step_sizes = [.1, .2, .4, .8]  # in units of max |Δ log K| per step
JJ = [J0]
J, G = J0, G0
for _ in range(nIter):
    d = -G / abs(G).max()  # the (normalized) descent direction
    trials = [misfit(logK + a*d) for a in step_sizes]
    best = int(np.argmin(trials))
    if trials[best] >= J:
        break  # no step size improves: stop
    logK = logK + step_sizes[best] * d
    J, G, fw = misfit(logK, gradient=True)
    JJ.append(J)
fw_final = water_cut(new_model(logK), new_model(logK).sim(dt, nSteps, S0, pbar=False)[0])
corr = np.corrcoef(-G0.ravel(), logK_true.ravel())[0, 1]

## Plot
fig, axs = freshfig("History-match gradient", ncols=3, nrows=2, figsize=(13, 8))

kws = dict(cmap="viridis", wells="color", finalize=False)
model.plt_field(axs[0, 0], logK_true, title="Truth, $\\log K$",
                levels=np.linspace(-3, 3, 19), cticks=np.arange(-3, 4), **kws)
m = abs(logK).max()  # NB: its own scale -- the update is far smaller than the truth
model.plt_field(axs[0, 2], logK, title=f"Estimate, after {len(JJ) - 1} descent steps",
                levels=np.linspace(-m, m, 19), cticks=[-m, 0, m], **kws)
m = abs(G0).max()
model.plt_field(axs[0, 1], G0, title="$∂J/∂\\log K$ at the prior ($\\log K = 0$)",
                cmap="RdBu_r", levels=np.linspace(-m, m, 21), cticks=[-m, 0, m],
                wells="color", finalize=False)
axs[0, 1].text(.02, .02, f"corr(−∂J/∂log K, truth) = {corr:.2f}",
               transform=axs[0, 1].transAxes, fontsize=8)

tt = dt * np.arange(1, nSteps + 1)
for ax, fw, title in [(axs[1, 0], fw_prior, "prior"), (axs[1, 1], fw_final, "estimate")]:
    for i, name in enumerate(producers):
        ax.plot(tt, obs[:, i], "*", c=f"C{i}", label=f"{name} obs.")
        ax.plot(tt, fw[:, i], "-", c=f"C{i}", label=f"{name} {title}")
    ax.set(title=f"Production history: {title}", xlabel="Time", ylabel="Water cut",
           ylim=(-.02, 1))
    ax.legend(loc="upper left", ncol=2, fontsize=8)

ax = axs[1, 2]
ax.semilogy(JJ, "o-")
ax.set(title="Misfit (MSE)", xlabel="Iteration")
ax.xaxis.get_major_locator().set_params(integer=True)

fig.tight_layout()

# Regression values, checked by `tests/test_examples.py`.
__digest__ = dict(misfit      = JJ,
                  gradient    = G0,
                  directional = [directional, fd],
                  corr        = [corr],
                  logK_final  = logK)

if __name__ == "__main__":
    show()
