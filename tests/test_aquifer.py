"""Tests of the aquifer contact, `aquifer_WI` (and the `aquifer` record key).

Structural properties, no figures: the half-cell transmissibility imposes the
aquifer pressure *at the face* (a 1D steady state is linear, and extrapolates
to `p_aq` at the boundary); a corner counts both of its faces, and a face to
an inactive cell counts as one to the outside; interior and inactive cells are
refused, as is a well model beside it; the influx is reported as any well's,
and supplies a lone producer even if incompressible; its markers can be hidden.
"""

import matplotlib.pyplot as plt
from typing import Any

import numpy as np
import pytest

from minires import ResSim, aquifer_WI


def test_pressure_is_imposed_at_the_face():
    """A row of cells, the aquifer at its west face, a producer at the east end.
    At the first step (`S = 0` so `λ_t = 1` throughout) the pressure profile
    is linear, dropping `q/T` per cell -- and `q/(2T)` from the face to the
    first centre: the half-cell transmissibility."""
    N, K, C, q, p_aq = 8, 2.5, .7, .3, 5.
    model = ResSim(Lx=N, Ly=1, Nx=N, Ny=1, K=K, cdarcy=C, wells=[
        dict(name="Aq", xy=[.5, .5], aquifer="W", bhp=p_aq),  # NB: not the N/S edges
        dict(name="P1", xy=[N - .5, .5], rate=-q)])
    T = C * K  # `hx = hy = 1`: the whole-cell transmissibility
    assert model.wells.WI[0] == 2 * T
    SS, PP = model.sim(.1, 3, np.zeros(N), pbar=False)
    expected = p_aq - q / (2 * T) - q / T * np.arange(N)
    assert np.allclose(PP[1], expected, rtol=0, atol=1e-12)
    # Incompressible: the aquifer supplies exactly what the producer takes
    assert np.allclose(model.wells.rates_by_well, [[q] * 3, [-q] * 3])
    # And what it supplies is water, appearing first in the contact cell
    assert SS[-1][0] > 0 and SS[-1][-1] == 0


def test_faces_are_counted():
    """Anisotropic `K` and cells tell the faces apart: `2 kx hy / hx` per
    x-face, `2 ky hx / hy` per y-face; a face to an inactive cell is one."""
    model = ResSim(Lx=1, Ly=2, Nx=4, Ny=4, cdarcy=1)
    model.K = np.stack([np.full(model.shape, 3.), np.full(model.shape, 5.)])
    model.active = np.arange(16).reshape(4, 4) != 10  # cell (2, 2) inactive
    hx, hy = model.hx, model.hy
    x_face, y_face = 2 * 3 * hy / hx, 2 * 5 * hx / hy  # 12 and 5

    def at(ix, iy):
        return [(ix + .5) * hx, (iy + .5) * hy]

    assert aquifer_WI(model, at(0, 1)) == x_face            # west edge
    assert aquifer_WI(model, at(1, 0)) == y_face            # south edge
    assert aquifer_WI(model, at(0, 0)) == x_face + y_face   # SW corner: both
    assert aquifer_WI(model, at(1, 2)) == x_face            # east face to (2, 2)
    assert aquifer_WI(model, at(2, 1)) == y_face            # north face to (2, 2)
    assert aquifer_WI(model, at(2, 3)) == y_face * 2        # north edge + (2, 2)
    # `faces` selects the directions that count
    assert aquifer_WI(model, at(0, 0), "W") == x_face       # the south edge sealed
    assert aquifer_WI(model, at(2, 3), "S") == y_face       # only the face to (2, 2)
    assert aquifer_WI(model, [at(3, 0), at(3, 3)], "EN").tolist() == [x_face, x_face + y_face]
    with pytest.raises(AssertionError, match="boundary"):
        aquifer_WI(model, at(1, 1))                         # interior
    with pytest.raises(AssertionError, match="boundary"):
        aquifer_WI(model, at(0, 1), "E")                    # no boundary face selected
    with pytest.raises(AssertionError, match="active"):
        aquifer_WI(model, at(2, 2))                         # inactive


def test_record_key():
    model = ResSim(Nx=4, Ny=4)
    with pytest.raises(AssertionError, match="no `rw`"):
        model.wells = [dict(xy=[0, 0], aquifer=True, bhp=1, rw=1e-3)]
    with pytest.raises(AssertionError, match="`xy`"):
        model.wells = [dict(path=[[0, 0], [0, 1]], aquifer=True, bhp=1, rw=1e-3)]
    # Several contacts form one well, sharing its name and pressure
    model.wells = [dict(name="Aq", xy=[[0, .1], [0, .5]], aquifer="W", bhp=[1, 2]),
                   dict(name="P1", xy=[1, 1], rate=-1)]
    wells: Any = model.wells  # (ty cannot see through the `__setattr__` normalization)
    assert wells.names == ["Aq", "P1"] and list(wells.group) == [0, 0, 1]
    assert np.array_equal(wells.bhp[:2], [[1, 2], [1, 2]])
    assert np.isfinite(wells.WI[:2]).all() and np.isnan(wells.WI[2])


def test_markers_can_be_hidden_and_faces_stroked():
    model = ResSim(Nx=4, Ny=4, wells=[
        dict(name="Aq", xy=[[0, .3], [0, .5], [0, .7]], aquifer=True, bhp=2),
        dict(name="P1", xy=[1, 1], rate=-1)])
    Z = np.zeros(model.Nxy)

    # The contact stroked: one segment per boundary face, W and S for cell (0, 0)
    fig, ax = plt.subplots()
    lc = model.plt_faces(ax, [[0, .5], [0, .7]], color="r")
    assert len(lc.get_segments()) == 2
    lc = model.plt_faces(ax, [[0, 0]])
    (W, S) = lc.get_segments()
    assert np.allclose(W, [[0, 0], [0, .25]]) and np.allclose(S, [[0, 0], [.25, 0]])
    assert len(model.plt_faces(ax, [[0, 0]], "N").get_segments()) == 0
    plt.close(fig)

    def n_scatters(**wells):
        fig, ax = plt.subplots()
        model.plt_field(ax, Z, finalize=False, wells=wells or True)
        n = len(ax.collections) - 1  # minus the field itself
        plt.close(fig)
        return n

    assert n_scatters() == 2                  # per sign: the producer, the (neutral) aquifer
    assert n_scatters(exclude=["Aq"]) == 1    # the aquifer's ring hidden
    assert n_scatters(exclude="P1") == 1      # a single name works too
    opts = dict(exclude=["Aq"], size=.5)
    n_scatters(**opts)
    assert "exclude" in opts, "the caller's dict must not be consumed"
