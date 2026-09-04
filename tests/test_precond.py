"""The cached-preconditioner pressure solve (`ResSim.cached_precond`).

It must reproduce the direct solve to the solver tolerance, reuse its
factorization across steps, refactorize (rather than fail) when the
system has drifted too far for the cached one to precondition it,
and leave the cache behind when the model is copied or pickled.

.. note:: It pays off unevenly, but never costs much.

    Where the saturation hardly moves -- a well test, primary depletion,
    pressure diffusion -- the pressure solve was the bulk of the cost
    (50--85%), and the run gets 2--6 times faster (`examples/buildup.py`:
    6x). In a waterflood, the explicit transport sub-stepping dominates
    -- 60% at $64^2$, 90% at $256^2$, its sub-step count growing as
    $N^2$ -- so the 3x on the solve amounts to 15--40% overall. Only a
    tiny system (`examples/buckley_leverett.py`, 1D) is quicker to
    factorize outright than to iterate on, by some 20%.

    The results differ from the direct solver's by the solver tolerance,
    four orders below what the regression tests and doctests resolve.
    The one place where the last bits *could* have mattered -- the CFL
    sub-step count of `saturation_step_upwind`, `ceil(dt * cfl1)`, which
    is discontinuous in the fluxes -- is kept off the round-off by
    `estimate_1CFL` (ref `tests/test_transport.py`).

    The cache is per instance (`_pLU`, some 1 MB at $64^2$) and is
    dropped on pickling and `deepcopy`, a `SuperLU` not being picklable
    -- which matters to the ensembles of HistoryMatching. It is a mere
    cache, so this is safe: a copy simply refactorizes on its next step.

    Also tried, and rejected: the same for the Newton solves of
    `saturation_step_implicit`. The upwind Jacobian is nearly triangular,
    so its factorization is nearly free, and it changes too much between
    Newton iterations for a stale one to precondition well: cached
    Krylov (BiCGSTAB, GMRES) was 1.5--30 times *slower* there.
    So that scheme keeps its `spsolve`.
    Likewise cheaper preconditioners than the exact LU (branch
    `cached-preconditioning`, `benchmarks/precond.py`): ILU (`spilu`)
    diverges from $256^2$ up, and an untuned algebraic multigrid
    (`pyamg`) overtakes the LU only at $512^2$ (4.6x) while needing
    ~50 iterations per solve on a 400x-contrast permeability -- i.e.
    losing on exactly the fields an ensemble draws.
"""

import numpy as np
import numpy.random as rnd
import pytest
from scipy.ndimage import uniform_filter as smooth

from TPFA_ResSim import ResSim

WELLS = [dict(xy=[0, 0], rate=+1), dict(xy=[1, 1], rate=-1)]


def model(vw=1.0, ct=0.0, N=32, seed=4, **kwargs):
    rnd.seed(seed)
    m = ResSim(Lx=1, Ly=1, Nx=N, Ny=N, vw=vw, ct=ct, wells=WELLS, **kwargs)
    m.K = np.exp(3 * smooth(smooth(rnd.randn(2, *m.shape))))
    return m


def count_factorizations(m, monkeypatch):
    """Instrument `splu` (as imported by the module) to count its calls."""
    import TPFA_ResSim

    calls = []
    orig = TPFA_ResSim.splu

    def counting(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(TPFA_ResSim, "splu", counting)
    return calls


@pytest.mark.parametrize("vw, ct", [(1.0, 0.0), (0.1, 0.0), (0.1, 1e-3)])
def test_agrees_with_direct(vw, ct):
    """Same trajectories, to well within the regression tests' `rtol=1e-4`."""
    S0 = np.zeros(32 * 32)
    P0 = np.ones(32 * 32) if ct else None
    kws = dict(dt=0.025, nSteps=28, S0=S0, P0=P0, pbar=False)
    S_dir, P_dir = model(vw, ct, cached_precond=False).sim(**kws)
    S_pcg, P_pcg = model(vw, ct).sim(**kws)
    assert np.allclose(S_pcg, S_dir, atol=1e-7)
    assert np.allclose(P_pcg, P_dir, rtol=1e-8, atol=1e-8 * np.abs(P_dir).max())


def test_factorization_is_reused(monkeypatch):
    """At unit viscosity ratio, one factorization serves the whole run."""
    m = model(cached_precond=True)
    calls = count_factorizations(m, monkeypatch)
    m.sim(0.025, 28, np.zeros(m.Nxy), pbar=False)
    assert len(calls) == 1
    # ... and the next run too: the cache is per-instance and persists.
    m.sim(0.025, 4, np.zeros(m.Nxy), pbar=False)
    assert len(calls) == 1


def test_refactorizes_when_stale(monkeypatch):
    """A preconditioner from another `K` is only a preconditioner:
    the result is still right, whether by convergence or by refactorizing."""
    m = model(cached_precond=True)
    calls = count_factorizations(m, monkeypatch)
    m.sim(0.025, 2, np.zeros(m.Nxy), pbar=False)
    assert len(calls) == 1
    # Drastically different permeability: the old factorization is useless.
    m.K = np.exp(8 * smooth(smooth(rnd.randn(2, *m.shape))))
    S_pcg, P_pcg = m.sim(0.025, 2, np.zeros(m.Nxy), pbar=False)
    assert len(calls) == 2
    m.cached_precond = False
    S_dir, P_dir = m.sim(0.025, 2, np.zeros(m.Nxy), pbar=False)
    assert np.allclose(P_pcg, P_dir, rtol=1e-8, atol=1e-8 * np.abs(P_dir).max())
    assert np.allclose(S_pcg, S_dir, atol=1e-7)


def test_copies_leave_the_cache_behind():
    """A `SuperLU` cannot be pickled; `deepcopy` and multiprocessing
    (both used by HistoryMatching) must not trip over the cache."""
    import copy
    import pickle

    m = model(cached_precond=True)
    S, P = m.sim(0.025, 2, np.zeros(m.Nxy), pbar=False)
    assert hasattr(m, "_pLU")
    for m2 in [copy.deepcopy(m), pickle.loads(pickle.dumps(m))]:
        assert not hasattr(m2, "_pLU")
        S2, P2 = m2.sim(0.025, 2, np.zeros(m.Nxy), pbar=False)  # refactorizes
        assert np.allclose(S2, S, atol=1e-7) and np.allclose(P2, P, rtol=1e-8)
    assert hasattr(m, "_pLU")  # the original keeps its cache


def test_direct_path(monkeypatch):
    """`cached_precond=False` factorizes afresh every step, and caches nothing."""
    m = model(cached_precond=False)
    calls = count_factorizations(m, monkeypatch)
    m.sim(0.025, 3, np.zeros(m.Nxy), pbar=False)
    assert len(calls) == 3 and not hasattr(m, "_pLU")
