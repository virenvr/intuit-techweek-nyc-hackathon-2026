"""Wilson score confidence intervals for binomial proportions."""

from __future__ import annotations

import numpy as np

from src.constants import INTERVAL_Z_SCORE


def wilson_score_interval(
    proportion: np.ndarray | float,
    n: int | np.ndarray | float,
    *,
    z: float = INTERVAL_Z_SCORE,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Wilson score interval for proportion p-hat = k/n.

    Closed form (proportion parameterization):

        [L, U] = (p-hat + z^2/(2n) +/- z*sqrt(p-hat*(1-p-hat)/n + z^2/(4n^2)))
                 / (1 + z^2/n)

    Recover k = p-hat * n when only the point estimate is available. The interval
    is algebraically inside [0, 1] without clamping.
    """
    p = np.clip(np.asarray(proportion, dtype=float), 0.0, 1.0)
    n_arr = np.maximum(np.asarray(n, dtype=float), 1.0)
    k = p * n_arr

    z2 = z * z
    denom = n_arr + z2
    center_num = k + z2 / 2.0
    radicand = k * (n_arr - k) / n_arr + z2 / 4.0
    root = z * np.sqrt(np.maximum(radicand, 0.0))

    lower = (center_num - root) / denom
    upper = (center_num + root) / denom

    # Guard against float noise; Wilson roots stay in [0, 1].
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
