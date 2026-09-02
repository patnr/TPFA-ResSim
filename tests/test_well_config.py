"""Tests of the well *grouping*, i.e. `ResSim.well_group` and `ResSim.well_names`.

The physics is tested elsewhere (`test_wells.py`, `test_compressible.py`, the
examples). What is at stake here is the bookkeeping: the model solves for
*completions* -- the rows of `well_xy`, `well_rates`, `actual_rates`, ... --
while a well may comprise several of them, which only the reporting need know.
"""

import numpy as np
import pytest

from TPFA_ResSim import ResSim


def a_model(**kwargs):
    """A 16² unit square, whose wells `kwargs` configures."""
    return ResSim(Lx=1, Ly=1, Nx=16, Ny=16, **kwargs)


def test_grouping_counts_wells_not_completions():
    """`nComp` indexes the arrays; `nWell` counts the wells that `well_group` says."""
    model = a_model(well_xy=[[0, 0], [0, 1], [1, 1]],
                    well_rates=[[.5], [.5], [-1]])
    assert (model.nComp, model.nWell) == (3, 3)   # ungrouped: a well per completion

    model.well_group = [0, 0, 1]                  # the two injectors are one well
    model.well_names = ["I1", "P1"]
    assert (model.nComp, model.nWell) == (3, 2)


def test_rates_by_well_sums_the_completions():
    """The point of the grouping: the reporting speaks of wells, the model of cells."""
    model = a_model(well_xy=[[0, 0], [0, 1], [1, 1]],
                    well_rates=[[.5], [.5], [-1]],
                    well_group=[0, 0, 1], well_names=["I1", "P1"])
    model.sim(.02, 5, model.swc*np.ones(model.Nxy), pbar=False)

    assert model.actual_rates.shape == (model.nComp, 5)
    assert model.rates_by_well.shape == (model.nWell, 5)
    assert np.allclose(model.rates_by_well, [5*[1], 5*[-1]])

    # Without any grouping it is a no-op
    plain = a_model(well_xy=[[0, 0], [1, 1]], well_rates=[[1], [-1]])
    plain.sim(.02, 3, plain.swc*np.ones(plain.Nxy), pbar=False)
    assert np.array_equal(plain.rates_by_well, plain.actual_rates)


def test_the_specs_must_match_the_completions():
    """Lest a spec left over from a previous configuration go unnoticed."""
    model = a_model(well_xy=[[0, 0], [1, 1]], well_rates=[[1], [-1]])
    model.well_xy = [[0, 0], [1, 1], [1, 0]]   # now the rates have too few rows
    with pytest.raises(AssertionError, match="3 completions"):
        model.sim(.02, 2, np.zeros(model.Nxy), pbar=False)


def test_well_markers_use_the_names():
    """The plot labels the completions by their well's name, when there is one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = a_model(well_xy=[[0, 0], [0, .2], [1, 1]],
                    well_rates=[[.5], [.5], [-1]],
                    well_group=[0, 0, 1], well_names=["I1", "P1"])
    _, ax = plt.subplots()
    try:
        model.plt_field(ax, np.zeros(model.Nxy), finalize=False)
        labels = sorted(t.get_text() for t in ax.texts)
    finally:
        plt.close("all")
    # I.e. one label per completion, but naming the well
    assert labels == ["I1", "I1", "P1"]
