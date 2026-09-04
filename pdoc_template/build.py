"""Build the docs: pdoc over the package *and* the examples, with the examples' figures.

Run it from the repo root: `uv run pdoc_template/build.py` (output in `docs/`). It is
`pdoc --math -t pdoc_template -o docs/ ./TPFA_ResSim ./examples` plus what the CLI cannot do:

- pdoc imports what it documents, so it would run the examples in any case. Running them
  here first lets us save the figures each one makes, so that its page can show them
  (pdoc itself cannot ship static files). The imported modules stay in `sys.modules`, so
  pdoc picks them up without running them again.
- The sidebar's "Contents" (the README's headings) goes one level deeper than pdoc's
  default of 2. That depth is set in a Python dict that neither the CLI nor the template
  can reach.
"""

import importlib
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")  # before pyplot gets imported (by the examples)
import matplotlib.pyplot as plt  # noqa: E402
import pdoc  # noqa: E402
import pdoc.render  # noqa: E402
import pdoc.render_helpers  # noqa: E402

here = Path(__file__).parent
root = here.parent
out = root / "docs"

sys.path.insert(0, str(root))  # makes `examples` importable

# Run the examples, saving each one's figures as `docs/examples/<name>-<i>.png`.
figures: dict[str, list[tuple[str, str]]] = {}  # module -> [(file, caption), ...]
for path in sorted((root / "examples").glob("[!_]*.py")):
    module = f"examples.{path.stem}"
    print(f"Running {module} ...", flush=True)
    plt.close("all")
    importlib.import_module(module)  # `__name__ != "__main__"` ⇒ no `show()`
    (out / "examples").mkdir(parents=True, exist_ok=True)
    figures[module] = []
    for i, num in enumerate(plt.get_fignums(), 1):
        fig = plt.figure(num)
        file = f"{path.stem}-{i}.png"
        fig.savefig(out / "examples" / file, dpi=120, bbox_inches="tight")
        caption = str(fig.get_label()) or f"Figure {i}"
        figures[module].append((file, caption))
plt.close("all")

# Render. The template (`module.html.jinja2`) reads `example_figures`.
toc = cast(dict, pdoc.render_helpers.markdown_extensions["toc"])
toc["depth"] = 3
pdoc.render.configure(math=True, template_directory=here)
template_globals: dict[str, Any] = pdoc.render.env.globals
template_globals["example_figures"] = figures
pdoc.pdoc(root / "TPFA_ResSim", root / "examples", output_directory=out)
