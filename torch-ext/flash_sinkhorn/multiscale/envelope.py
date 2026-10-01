"""Two warning thresholds of the multiscale solver, read from the geometry alone: the width of the coarse cells in
blurs (CELL_EDGE_BLURS) and the fp32 ulp of the largest squared norm over eps (PRECISION_LIMIT). Crossing either
warns; the solver still runs, and neither number alone decides convergence or failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .geometry import CELL_EDGE_BLURS, JointSort, cell_edge, narrowest_cell_edge

PRECISION_LIMIT = 0.05


@dataclass(frozen=True)
class Envelope:
    """``cell_blurs``, the longest edge of the coarse cells in blurs ``sqrt(eps)``; ``narrowest_blurs``, the
    narrowest cells this GPU's cell budget allows, in blurs; ``ulp_over_eps``, the fp32 ulp of the largest
    squared norm ``|x|^2`` of either cloud, over eps."""
    cell_blurs: float
    narrowest_blurs: float
    ulp_over_eps: float

    @property
    def reasons(self) -> List[str]:
        """One sentence per limit the problem exceeds, with the factor the blur would have to grow by."""
        out = []
        if self.cell_blurs > CELL_EDGE_BLURS:
            out.append(f"the coarse cells are {self.cell_blurs:.3g} blurs wide, above the warning threshold "
                       f"{CELL_EDGE_BLURS:g}; on this GPU a level meets it from a blur "
                       f"{self.narrowest_blurs / CELL_EDGE_BLURS:.3g} times the current one")
        if self.ulp_over_eps > PRECISION_LIMIT:
            out.append(f"ulp(|x|^2)/eps is {self.ulp_over_eps:.2g}, above the warning threshold {PRECISION_LIMIT:g}; "
                       f"a blur {math.sqrt(self.ulp_over_eps / PRECISION_LIMIT):.3g} times larger brings it there")
        return out

    @property
    def inside(self) -> bool:
        return not self.reasons


def envelope(joint: JointSort, counts_x, counts_y, B: int, t: int, l2_bytes: int, eps: float) -> Envelope:
    """The envelope of the sorted clouds at cell level ``t``."""
    blur = math.sqrt(eps)
    norm = max(float(p.points.double().square().sum(1).max()) for p in (joint.source, joint.target))
    ulp = 2.0 ** (math.floor(math.log2(norm)) - 23) if norm > 0 else 0.0
    narrowest = narrowest_cell_edge(counts_x, counts_y, joint.source.n, joint.target.n, B, l2_bytes, joint.sides)
    return Envelope(cell_edge(joint.sides, t) / blur, narrowest / blur, ulp / eps)
