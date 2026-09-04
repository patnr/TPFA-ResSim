"""Build the docs: what `pdoc --math -t pdoc_template -o docs/ ./TPFA_ResSim` does, except
that the sidebar's "Contents" (the README's headings) goes one level deeper than pdoc's
default of 2. That depth is set in a Python dict that neither the CLI nor the template
can reach, hence this script. Run it from the repo root: `uv run pdoc_template/build.py`.
"""

from pathlib import Path

import pdoc
import pdoc.render
import pdoc.render_helpers

here = Path(__file__).parent
pdoc.render_helpers.markdown_extensions["toc"]["depth"] = 3
pdoc.render.configure(math=True, template_directory=here)
pdoc.pdoc(here.parent / "TPFA_ResSim", output_directory=here.parent / "docs")
