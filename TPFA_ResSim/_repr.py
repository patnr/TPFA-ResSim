"""The `repr` shared by `TPFA_ResSim.ResSim` and `TPFA_ResSim.wells.Wells`."""

import numpy as np


class AlignedRepr:
    """Provides `__repr__` listing the object's (public) attributes, aligned.

    Arrays are summarized (rather than dumped in full), and nested `repr`s
    (such as that of the wells within the model) are indented to their key.
    """

    def __repr__(self) -> str:
        keys = [k for k in vars(self) if not k.startswith("_")]
        width = max(map(len, keys), default=0)
        hang = "\n" + " " * (width + 4)
        with np.printoptions(threshold=50, edgeitems=2, linewidth=70):
            items = "".join(
                f"\n  {k:>{width}}: " + repr(getattr(self, k)).replace("\n", hang)
                for k in keys
            )
        items = "\n".join(line.rstrip() for line in items.split("\n"))
        return f"{type(self).__name__}({items}\n)"
