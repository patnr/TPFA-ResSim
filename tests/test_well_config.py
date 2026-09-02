"""Tests of the well *configuration*, i.e. `Wells` and the grouping.

The physics is tested elsewhere (`test_wells.py`, `test_compressible.py`, the
examples). What is at stake here is the bookkeeping: that a list of records
becomes the flat, per-completion arrays that the model runs on -- `Wells.xy`,
`Wells.rates`, `Wells.bhp`, `Wells.WI` -- exactly as a hand-written config would,
and that the grouping (`Wells.group`, `Wells.names`) recovers the wells from the
completions for the reporting.
"""

from typing import Any

import numpy as np
import pytest

from TPFA_ResSim import ResSim, Wells, peaceman_WI, well_path

rw = 1e-3


def a_model(**kwargs):
    """A 16² unit square, whose wells `kwargs` configures."""
    return ResSim(Lx=1, Ly=1, Nx=16, Ny=16, **kwargs)


def snap(model, xy):
    """The grid nodes that the wells get collocated with, ref `Wells.xy`."""
    xy = np.asarray(xy, float).reshape((-1, 2))
    return np.array([model.ind2xy(model.xy2ind(x, y)) for x, y in xy])


def test_wells_assembles_the_arrays():
    """The concise case: a position and a (signed) rate is a well."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])

    assert np.array_equal(model.wells.xy, snap(model, [[0, 0], [1, 1]]))
    assert np.array_equal(model.wells.rates, [[1], [-1]])  # NB: still a singleton
    assert model.wells.bhp is None   # no well asked for BHP control
    assert model.wells.WI is None    # ... nor for a well model
    # No grouping to speak of: one well per completion, named by their index
    assert (model.nComp, model.wells.nWell) == (2, 2)
    assert model.wells.names == ["0", "1"]
    assert np.array_equal(model.wells.group, [0, 1])


def test_wells_is_equivalent_to_a_hand_set_config():
    """The helper is sugar: it must produce the very same run, bit for bit."""
    def run(model):
        return model.sim(.05, 10, model.swc*np.ones(model.Nxy), pbar=False)

    by_hand = a_model(wells=Wells(xy=[[0, 0], [1, 1]], rates=[[1], [-1]]))
    by_hand.wells.WI = [np.nan, peaceman_WI(by_hand, [[1, 1]], rw)[0]]
    by_helper = a_model(wells=[dict(xy=[0, 0], rate=+1),
                               dict(xy=[1, 1], rate=-1, rw=rw)])

    assert np.array_equal(by_hand.wells.WI, by_helper.wells.WI, equal_nan=True)
    for a, b in zip(run(by_hand), run(by_helper)):
        assert np.array_equal(a, b)
    assert np.array_equal(by_hand.wells.actual_bhp, by_helper.wells.actual_bhp, equal_nan=True)


def test_wells_names_the_wells():
    """Explicitly, or by the keys of a `dict` of records, else by their index."""
    listed = a_model(wells=[dict(name="I1", xy=[0, 0], rate=+1),
                            dict(xy=[1, 1], rate=-1)])
    assert listed.wells.names == ["I1", "1"]

    keyed = a_model(wells={"I1": dict(xy=[0, 0], rate=+1),
                           "P1": dict(xy=[1, 1], rate=-1)})
    assert keyed.wells.names == ["I1", "P1"]
    assert np.array_equal(keyed.wells.rates, listed.wells.rates)


def test_wells_computes_the_well_index():
    """`rw` (and `skin`) invoke `peaceman_WI`; `WI` bypasses it; neither ⇒ `nan`."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-.5, rw=rw),
                           dict(xy=[1, 0], rate=-.5, rw=rw, skin=2.)])
    assert np.isnan(model.wells.WI[0])                      # no well model
    assert model.wells.WI[1] == peaceman_WI(model, [[1, 1]], rw)[0]
    assert model.wells.WI[2] == peaceman_WI(model, [[1, 0]], rw, skin=2.)[0]

    given = a_model(wells=[dict(xy=[0, 0], rate=+1, WI=3., rw=rw),
                           dict(xy=[1, 1], rate=-1)])
    assert given.wells.WI[0] == 3.   # `WI` wins over `rw`


def test_wells_discretizes_a_path():
    """A `path` becomes several completions of a single (named, grouped) well."""
    proto = a_model()
    xy, WI, alloc = well_path(proto, [[0, 0], [0, 1]], rw)

    model = a_model(wells=[
        dict(name="I1", path=[[0, 0], [0, 1]], rate=+1, rw=rw),
        dict(name="P1", xy=[1, 1], rate=-1),
    ])

    assert (model.nComp, model.wells.nWell) == (len(xy) + 1, 2)
    assert model.wells.names == ["I1", "P1"]
    assert np.array_equal(model.wells.group, len(xy)*[0] + [1])
    assert np.array_equal(model.wells.xy, [*xy, *snap(model, [1, 1])])
    assert np.array_equal(model.wells.WI, np.append(WI, np.nan), equal_nan=True)
    # The rate is apportioned by well index, as `well_path` prescribes
    assert np.allclose(model.wells.rates[:-1, 0], alloc)
    assert np.isclose(model.wells.rates[:, 0].sum(), 0)


def test_wells_apportions_among_several_xy():
    """Several `xy` likewise compose one well -- uniformly, absent well indices."""
    model = a_model(wells=[dict(xy=[[0, 0], [0, 1]], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    assert (model.nComp, model.wells.nWell) == (3, 2)
    assert np.array_equal(model.wells.group, [0, 0, 1])
    assert np.allclose(model.wells.rates[:, 0], [.5, .5, -1])


def test_wells_shares_the_bhp_among_the_completions():
    """As a wellbore does -- whereas the rate gets split. Ref `well_path`."""
    model = a_model(wells=[dict(path=[[0, 0], [0, 1]], bhp=3., rw=rw),
                           dict(xy=[1, 1], rate=-1)])
    nc = int((model.wells.group == 0).sum())
    assert np.all(model.wells.bhp[:nc] == 3.)
    assert np.isnan(model.wells.bhp[nc:]).all()      # the producer: rate-controlled
    assert np.all(model.wells.rates[:nc] == 0)       # the (ignored) fill value


def test_wells_broadcasts_the_constant_schedules():
    """A scalar rate joins a schedule; a spec no well varies stays a singleton."""
    schedule = [1, 2, 2]
    model = a_model(wells=[dict(xy=[0, 0], rate=schedule),
                           dict(xy=[1, 1], rate=-1)])
    assert np.array_equal(model.wells.rates, [schedule, 3*[-1]])

    const = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    assert const.wells.rates.shape == (2, 1)


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
    assert np.array_equal(model.wells.rates, [[0]])


def test_wells_validates_the_records():
    with pytest.raises(TypeError, match="Unknown key"):
        a_model(wells=[dict(xy=[0, 0], rates=1)])       # sic: `rate`
    with pytest.raises(AssertionError, match="`xy` or `path`"):
        a_model(wells=[dict(xy=[0, 0], path=[[0, 0], [0, 1]], rate=1, rw=rw)])
    with pytest.raises(AssertionError, match="requires `rw`"):
        a_model(wells=[dict(path=[[0, 0], [0, 1]], rate=1)])
    with pytest.raises(AssertionError, match="an `xy`"):
        a_model(wells=[dict(rate=1)])


def test_assigning_nothing_empties_the_wells():
    model = a_model(wells=[dict(xy=[0, 0], rate=0)])
    for nothing in [[], None]:
        model.wells = nothing
        assert model.nComp == 0     # i.e. `Wells.xy` is empty, shape `(0, 2)`
        for key in ["rates", "bhp", "WI", "group", "names"]:
            assert getattr(model.wells, key) is None


def test_the_records_are_read_not_kept():
    """The arrays being the whole of the configuration, the records are inert --
    and must be left untouched, the caller being free to reuse them."""
    records = [dict(xy=[0, 0], rate=+1), dict(name="P1", xy=[1, 1], rate=-1)]
    a_model(wells=records)
    assert records == [dict(xy=[0, 0], rate=+1),
                       dict(name="P1", xy=[1, 1], rate=-1)]


def test_assigning_the_records_reconfigures():
    """Assignment is what applies them -- afresh, whatever was there before."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    # NB: `Any`, lest `ty` take the reads below to be of the records themselves
    records: Any = [dict(name="P1", xy=[1, 1], rate=-1, rw=rw)]
    model.wells = records

    assert model.nComp == 1
    assert model.wells.names == ["P1"]
    assert np.array_equal(model.wells.rates, [[-1]])
    assert np.isfinite(model.wells.WI).all()


def test_editing_the_arrays_needs_no_bookkeeping():
    """There being a single representation, an edit cannot outdate another one.

    (Which the records, were they retained alongside, otherwise would be.)
    """
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    model.wells.rates = [[2], [-2]]             # as an ensemble/EnOpt loop does
    assert np.array_equal(model.wells.rates, [[2], [-2]])   # the edit stands
    assert model.wells.names == ["0", "1"]      # as does the rest of the config


def test_arrays_remain_writable():
    """Ensemble methods perturb the arrays directly; the helper must not preclude it."""
    model = a_model(wells=[dict(xy=[0, 0], rate=+1),
                           dict(xy=[1, 1], rate=-1)])
    model.wells.xy = [[0, .5], [1, 1]]         # as EnOpt over positions would
    model.wells.rates = [[2], [-2]]
    assert np.array_equal(model.wells.xy, snap(model, [[0, .5], [1, 1]]))
    ctrl = model.well_controls(None, None, 0)
    assert np.array_equal(ctrl["rates"], [2, -2])   # i.e. the specs got re-read


def test_grouping_counts_wells_not_completions():
    """`nComp` indexes the arrays; `nWell` counts the wells that `Wells.group` says."""
    model = a_model(wells=Wells(xy=[[0, 0], [0, 1], [1, 1]],
                                rates=[[.5], [.5], [-1]]))
    assert (model.nComp, model.wells.nWell) == (3, 3)   # ungrouped: a well per completion

    model.wells.group = [0, 0, 1]                  # the two injectors are one well
    model.wells.names = ["I1", "P1"]
    assert (model.nComp, model.wells.nWell) == (3, 2)


def test_rates_by_well_sums_the_completions():
    """The point of the grouping: the reporting speaks of wells, the model of cells."""
    model = a_model(wells=Wells(xy=[[0, 0], [0, 1], [1, 1]],
                                rates=[[.5], [.5], [-1]],
                                group=[0, 0, 1], names=["I1", "P1"]))
    model.sim(.02, 5, model.swc*np.ones(model.Nxy), pbar=False)

    assert model.wells.actual_rates.shape == (model.nComp, 5)
    assert model.wells.rates_by_well.shape == (model.wells.nWell, 5)
    assert np.allclose(model.wells.rates_by_well, [5*[1], 5*[-1]])

    # Without any grouping it is a no-op
    plain = a_model(wells=Wells(xy=[[0, 0], [1, 1]], rates=[[1], [-1]]))
    plain.sim(.02, 3, plain.swc*np.ones(plain.Nxy), pbar=False)
    assert np.array_equal(plain.wells.rates_by_well, plain.wells.actual_rates)


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
    ctrl = a_model(wells=Wells(xy=[[0, 0]])).well_controls(None, None, 7)
    assert ctrl["rates"] == [0] and np.isnan(ctrl["bhp"]).all()


def test_the_specs_must_match_the_completions():
    """Lest a spec left over from a previous configuration go unnoticed."""
    model = a_model(wells=Wells(xy=[[0, 0], [1, 1]], rates=[[1], [-1]]))
    model.wells.xy = [[0, 0], [1, 1], [1, 0]]   # now the rates have too few rows
    with pytest.raises(AssertionError, match="3 completions"):
        model.sim(.02, 2, np.zeros(model.Nxy), pbar=False)


def test_well_markers_use_the_names():
    """The plot labels the completions by their well's name, when there is one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = a_model(wells=Wells(xy=[[0, 0], [0, .2], [1, 1]],
                                rates=[[.5], [.5], [-1]],
                                group=[0, 0, 1], names=["I1", "P1"]))
    _, ax = plt.subplots()
    try:
        model.plt_field(ax, np.zeros(model.Nxy), finalize=False)
        labels = sorted(t.get_text() for t in ax.texts)
    finally:
        plt.close("all")
    # I.e. one label per completion, but naming the well
    assert labels == ["I1", "I1", "P1"]
