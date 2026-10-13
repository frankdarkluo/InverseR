from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from inverser.schemas import Layout


@dataclass(frozen=True)
class EndpointScore:
    success: bool
    max_position_error_m: float | None
    failure: str | None


def bottleneck_distance(
    predicted: list[tuple[float, float]], target: list[tuple[float, float]]
) -> float:
    if len(predicted) != len(target):
        raise ValueError("Matching requires equal-size point sets")
    if not predicted:
        return 0.0
    distances = np.linalg.norm(
        np.asarray(predicted)[:, None, :] - np.asarray(target)[None, :, :], axis=-1
    )
    thresholds = np.unique(distances)
    lo, hi = 0, len(thresholds) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        matching = maximum_bipartite_matching(
            csr_matrix(distances <= thresholds[mid]), perm_type="column"
        )
        if np.all(matching >= 0):
            hi = mid
        else:
            lo = mid + 1
    return float(thresholds[lo])


def score_endpoint(
    predicted: Layout, target: Layout, tolerance_m: float, *, no_scratch: bool = True
) -> EndpointScore:
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Tolerance must be a positive finite distance")
    if any(ball.status == "unobserved" for ball in (*predicted.balls, *target.balls)):
        raise ValueError("Unobserved balls cannot be scored as fully known endpoints")
    if no_scratch and any(
        ball.kind == "cue" and ball.status == "pocketed" for ball in predicted.balls
    ):
        return EndpointScore(False, None, "scratch")

    def inventory(layout: Layout) -> Counter[tuple[str, str]]:
        return Counter((ball.kind, ball.status) for ball in layout.balls)

    if inventory(predicted) != inventory(target):
        return EndpointScore(False, None, "inventory_or_pocket_status_mismatch")
    error = 0.0
    for kind in {ball.kind for ball in target.balls}:
        p = [ball.xy_m for ball in predicted.balls if ball.kind == kind and ball.xy_m is not None]
        t = [ball.xy_m for ball in target.balls if ball.kind == kind and ball.xy_m is not None]
        error = max(error, bottleneck_distance(p, t))
    return EndpointScore(error <= tolerance_m, error, None if error <= tolerance_m else "position")
