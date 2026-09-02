"""Tests of the well *configuration*, i.e. `ResSim.wells` and the grouping.

The physics is tested elsewhere (`test_wells.py`, `test_compressible.py`, the
examples). What is at stake here is the bookkeeping: that a list of records
becomes the flat, per-completion arrays that the model runs on -- `well_xy`,
`well_rates`, `well_bhp`, `well_WI` -- exactly as a hand-written config would,
and that the grouping (`well_group`, `well_names`) recovers the wells from the
completions for the reporting.
"""

import numpy as np
import pytest

from TPFA_ResSim import ResSim

rw = 1e-3


def a_model(**kwargs):
    """A 16² unit square, whose wells `kwargs` configures."""
    return ResSim(Lx=1, Ly=1, Nx=16, Ny=16, **kwargs)


def snap(model, xy):
    """The grid nodes that the wells get collocated with, ref `ResSim.well_xy`."""
    xy = np.asarray(xy, float).reshape((-1, 2))
    return np.array([model.ind2xy(model.xy2ind(x, y)) for x, y in xy])


def test_wells_assembles_the_arrays():
    """The concise case: a position and a (signed) rate is a well."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])

    assert np.array_equal(model.well_xy, snap(model, [[0, 0], [1, 1]]))
    assert np.array_equal(model.well_rates, [[1], [-1]])  # NB: still a singleton
    assert model.well_bhp is None   # no well asked for BHP control
    assert model.well_WI is None    # ... nor for a well model
    # No grouping to speak of: one well per completion, named by their index
    assert (model.nComp, model.nWell) == (2, 2)
    assert model.well_names == ["0", "1"]
    assert np.array_equal(model.well_group, [0, 1])


def test_wells_is_equivalent_to_a_hand_set_config():
    """The helper is sugar: it must produce the very same run, bit for bit."""
    def run(model):
        return model.sim(.05, 10, model.swc*np.ones(model.Nxy), pbar=False)

    by_hand = a_model(well_xy=[[0, 0], [1, 1]], well_rates=[[1], [-1]])
    by_hand.well_WI = [np.nan, by_hand.peaceman_WI([[1, 1]], rw)[0]]
    by_helper = a_model(wells=[dict(xy=[0, 0], rate=+1),
                               dict(xy=[1, 1], rate=-1, rw=rw)])

    assert np.array_equal(by_hand.well_WI, by_helper.well_WI, equal_nan=True)
    for a, b in zip(run(by_hand), run(by_helper)):
        assert np.array_equal(a, b)
    assert np.array_equal(by_hand.actual_bhp, by_helper.actual_bhp, equal_nan=True)


def test_wells_names_the_wells():
    """Explicitly, or by the keys of a `dict` of records, else by their index."""
    listed = a_model(wells=[dict(name="I1", xy=[0, 0], rate=+1),
                            dict(xy=[1, 1], rate=-1)])
    assert listed.well_names == ["I1", "1"]

    keyed = a_model(wells={"I1": dict(xy=[0, 0], rate=+1),
                           "P1": dict(xy=[1, 1], rate=-1)})
    assert keyed.well_names == ["I1", "P1"]
    assert np.array_equal(keyed.well_rates, listed.well_rates)


def test_wells_computes_the_well_index():
    """`rw` (and `skin`) invoke `peaceman_WI`; `WI` bypasses it; neither ⇒ `nan`."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-.5, rw=rw),
                           dict(xy=[1, 0], rate=-.5, rw=rw, skin=2.)])
    assert np.isnan(model.well_WI[0])                      # no well model
    assert model.well_WI[1] == model.peaceman_WI([[1, 1]], rw)[0]
    assert model.well_WI[2] == model.peaceman_WI([[1, 0]], rw, skin=2.)[0]

    given = a_model(wells=[dict(xy=[0, 0], rate=+1, WI=3., rw=rw),
                           dict(xy=[1, 1], rate=-1)])
    assert given.well_WI[0] == 3.   # `WI` wins over `rw`


def test_wells_discretizes_a_path():
    """A `path` becomes several completions of a single (named, grouped) well."""
    proto = a_model()
    xy, WI, alloc = proto.well_path([[0, 0], [0, 1]], rw)

    model = a_model(wells=[
        dict(name="I1", path=[[0, 0], [0, 1]], rate=+1, rw=rw),
        dict(name="P1", xy=[1, 1], rate=-1),
    ])

    assert (model.nComp, model.nWell) == (len(xy) + 1, 2)
    assert model.well_names == ["I1", "P1"]
    assert np.array_equal(model.well_group, len(xy)*[0] + [1])
    assert np.array_equal(model.well_xy, [*xy, *snap(model, [1, 1])])
    assert np.array_equal(model.well_WI, np.append(WI, np.nan), equal_nan=True)
    # The rate is apportioned by well index, as `well_path` prescribes
    assert np.allclose(model.well_rates[:-1, 0], alloc)
    assert np.isclose(model.well_rates[:, 0].sum(), 0)


def test_wells_apportions_among_several_xy():
    """Several `xy` likewise compose one well -- uniformly, absent well indices."""
    model = a_model(wells=[dict(xy=[[0, 0], [0, 1]], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    assert (model.nComp, model.nWell) == (3, 2)
    assert np.array_equal(model.well_group, [0, 0, 1])
    assert np.allclose(model.well_rates[:, 0], [.5, .5, -1])


def test_wells_shares_the_bhp_among_the_completions():
    """As a wellbore does -- whereas the rate gets split. Ref `well_path`."""
    model = a_model(wells=[dict(path=[[0, 0], [0, 1]], bhp=3., rw=rw),
                           dict(xy=[1, 1], rate=-1)])
    nc = int((model.well_group == 0).sum())
    assert np.all(model.well_bhp[:nc] == 3.)
    assert np.isnan(model.well_bhp[nc:]).all()      # the producer: rate-controlled
    assert np.all(model.well_rates[:nc] == 0)       # the (ignored) fill value


def test_wells_broadcasts_the_constant_schedules():
    """A scalar rate joins a schedule; a spec no well varies stays a singleton."""
    schedule = [1, 2, 2]
    model = a_model(wells=[dict(xy=[0, 0], rate=schedule),
                           dict(xy=[1, 1], rate=-1)])
    assert np.array_equal(model.well_rates, [schedule, 3*[-1]])

    const = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    assert const.well_rates.shape == (2, 1)


def test_wells_rejects_unequal_schedules():
    with pytest.raises(AssertionError, match="equal length"):
        a_model(wells=[dict(xy=[0, 0], rate=[1, 1, 1]),
                       dict(xy=[1, 1], rate=[-1, -1])])


def test_wells_insists_on_a_control():
    """An uncontrolled well would silently be a shut one, so it is an error."""
    with pytest.raises(AssertionError, match="no control"):
        a_model(wells=[dict(xy=[0, 0])])
    # Whereas shutting it in is fine -- as long as it is said
    model = a_model(wells=[dict(xy=[0, 0], rate=0)])
    assert np.array_equal(model.well_rates, [[0]])


def test_wells_validates_the_records():
    with pytest.raises(TypeError, match="Unknown key"):
        a_model(wells=[dict(xy=[0, 0], rates=1)])       # sic: `rate`
    with pytest.raises(AssertionError, match="`xy` or `path`"):
        a_model(wells=[dict(xy=[0, 0], path=[[0, 0], [0, 1]], rate=1, rw=rw)])
    with pytest.raises(AssertionError, match="requires `rw`"):
        a_model(wells=[dict(path=[[0, 0], [0, 1]], rate=1)])
    with pytest.raises(AssertionError, match="an `xy`"):
        a_model(wells=[dict(rate=1)])


def test_the_records_clear():
    model = a_model(wells=[dict(xy=[0, 0], rate=0)])
    model.wells = []
    for key in ["well_xy", "well_rates", "well_bhp",
                "well_WI", "well_group", "well_names"]:
        assert getattr(model, key) is None


def test_the_records_are_readable_and_unmutated():
    records = [dict(xy=[0, 0], rate=+1), dict(name="P1", xy=[1, 1], rate=-1)]
    model = a_model(wells=records)
    assert model.wells == records
    assert records[1] == dict(name="P1", xy=[1, 1], rate=-1)   # i.e. not mutated


def test_assigning_the_records_reconfigures():
    """Assignment is what applies them -- afresh, whatever was there before."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    model.wells = [dict(name="P1", xy=[1, 1], rate=-1, rw=rw)]

    assert model.nComp == 1
    assert model.well_names == ["P1"]
    assert np.array_equal(model.well_rates, [[-1]])
    assert np.isfinite(model.well_WI).all()


def test_editing_the_arrays_outdates_the_records():
    """So that `wells`, whenever it holds anything, describes the current wells."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    assert model.wells is not None
    model.well_rates = [[2], [-2]]             # as an ensemble/EnOpt loop does
    assert model.wells is None                 # ... the records no longer describe it
    assert np.array_equal(model.well_rates, [[2], [-2]])    # but the edit stands
    assert model.well_names == ["0", "1"]      # as does the rest of the config


def test_the_documented_examples_hold():
    """The worked examples of `ResSim.wells` -- an attribute docstring, so no doctest."""
    model = ResSim(Lx=1, Ly=1, Nx=16, Ny=16, wells={
        "I1": dict(xy=[0, 0], rate=+1),
        "P1": dict(xy=[1, 1], rate=-1, rw=1e-3),
    })
    assert model.well_names == ["I1", "P1"]
    assert np.array_equal(model.well_rates, [[1], [-1]])
    assert np.array_equal(model.well_WI.round(3), [np.nan, 2.498], equal_nan=True)

    model = ResSim(Lx=1, Ly=1, Nx=10, Ny=10, wells=[
        dict(name="I1", path=[[.05, .05], [.45, .05]], rate=+1, rw=1e-2),
        dict(name="P1", xy=[.95, .95], rate=-1),
    ])
    assert (model.nComp, model.nWell) == (6, 2)
    assert np.array_equal(model.well_group, [0, 0, 0, 0, 0, 1])
    assert np.array_equal(model.well_rates.ravel().round(3),
                          [.125, .25, .25, .25, .125, -1])


def test_arrays_remain_writable():
    """Ensemble methods perturb the arrays directly; the helper must not preclude it."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    model.well_xy = [[0, .5], [1, 1]]         # as EnOpt over positions would
    model.well_rates = [[2], [-2]]
    assert np.array_equal(model.well_xy, snap(model, [[0, .5], [1, 1]]))
    ctrl = model.well_controls(None, None, 0)
    assert np.array_equal(ctrl["rates"], [2, -2])   # i.e. the specs got re-read


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


def test_the_specs_are_read_per_step():
    """`well_controls` looks the specs up at each `k`; a singleton stays constant."""
    model = a_model(wells=[dict(xy=[0, 0], rate=[1, 2, 2]),
                           dict(xy=[1, 1], bhp=.5, rw=rw)])
    ctrl = [model.well_controls(None, None, k) for k in range(3)]

    assert [c["rates"][0] for c in ctrl] == [1, 2, 2]        # the schedule
    assert [c["rates"][1] for c in ctrl] == [0, 0, 0]        # the BHP well's fill
    for c in ctrl:                                           # the singleton BHP
        assert np.array_equal(c["bhp"], [np.nan, .5], equal_nan=True)

    # An unset spec is substituted: `0` for the rates, `nan` (⇒ rate-ctrl) for BHP
    ctrl = a_model(well_xy=[[0, 0]]).well_controls(None, None, 7)
    assert ctrl["rates"] == [0] and np.isnan(ctrl["bhp"]).all()


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
