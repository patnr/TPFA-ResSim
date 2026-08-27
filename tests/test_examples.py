"""Run each script of `examples/`, and check its output against `references.py`.

The scripts in `examples/` are the primary illustrations of this package
(they are what produces `collage.jpg`). Since they are also its most
"full-fledged" usages, they double as its regression tests: here we simply
run them all, and verify that they still produce the values that they used to.

The plotting is *included* in this (it is the part most likely to break with
new matplotlib versions), but the figures are neither displayed nor saved:
the scripts guard their call to `show()` with `if __name__ == "__main__"`,
which is False when they are run (with `runpy`) from here.

The values are compared with `rtol=1e-4`, which is far above the noise of the
(direct) linear solvers, but allows for storing them with few decimals.
"""

import runpy
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Prevent the examples' figures from popping up
import matplotlib.pyplot as plt  # noqa: E402

from references import references  # noqa: E402

examples = sorted(Path(__file__).parents[1].joinpath("examples").glob("[!_]*.py"))
assert examples, "No examples found!"


def fingerprint(x, n=8):
    """Subsample `x` (flattened) to at most `n` values, spanning it entirely."""
    x = np.asarray(x, dtype=float).ravel()
    return x[np.linspace(0, x.size - 1, min(n, x.size), dtype=int)]


def digest(namespace):
    """Extract the values that an example wants checked.

    An example declares them by `__digest__` (a dict of arrays).
    Failing that, we fall back on its saturation/pressure trajectories.
    """
    values = namespace.get("__digest__")
    if values is None:
        values = {k: namespace[k] for k in ["SS", "PP"] if k in namespace}
    assert values, "The example yielded no values to check. Set `__digest__`."
    return {k: fingerprint(v) for k, v in values.items()}


def run(path):
    """Execute the example (`run_name != "__main__"` ⇒ no `show()`)."""
    try:
        return digest(runpy.run_path(str(path)))
    finally:
        plt.close("all")


def as_source(name, values):
    """Render `values` as a (paste-able) entry of `references.py`."""
    fmt = lambda v: "[" + ", ".join(f"{x:.6g}" for x in v) + "]"
    entries = "\n".join(f"        {k} = {fmt(v)}," for k, v in values.items())
    return f'    "{name}": dict(\n{entries}\n    ),'


@pytest.mark.parametrize("path", examples, ids=lambda p: p.stem)
def test_example(path):
    """Run the example; compare with the stored reference values."""
    values = run(path)
    expected = references[path.stem]

    hint = ("\nIf the change is intentional, update `tests/references.py` to:\n\n"
            + as_source(path.stem, values))
    assert set(values) == set(expected), "Keys of `__digest__` changed." + hint
    for k, v in values.items():
        assert np.isclose(v, expected[k], rtol=1e-4).all(), (
            f"Example '{path.stem}' produced other values for '{k}':"
            f"\n  now: {v}\n  ref: {np.asarray(expected[k])}" + hint)


def test_references_are_current():
    """No references for examples that no longer exist."""
    assert set(references) == {p.stem for p in examples}


if __name__ == "__main__":
    # Regenerate (print) the entire reference table
    print("references = {")
    for path in examples:
        print(as_source(path.stem, run(path)))
    print("}")
