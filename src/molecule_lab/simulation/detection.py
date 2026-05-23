"""Bond breaking detection."""

from __future__ import annotations

import math

import numpy as np


def detect_broken_bonds(
    pos: np.ndarray,
    bonds: list[tuple[int, int, float, float, float]],
    break_frac: float,
    break_distance_factor: float,
) -> list[dict[str, float | int | str]]:
    """Detect bonds that have become chemically implausibly stretched.

    The original criterion used only V > break_frac * D_e. With a Morse
    potential, using a high fraction such as 0.95 detects rupture only at very
    large separations. This detector combines an energy criterion with a
    distance criterion, which makes the result less dependent on the asymptotic
    tail of the Morse potential.
    """
    broken: list[dict[str, float | int | str]] = []
    for idx, (i, j, r0, de, alpha) in enumerate(bonds):
        r = np.linalg.norm(pos[i] - pos[j])
        if r <= r0:
            continue

        u = math.exp(-alpha * (r - r0))
        potential = de * (1 - u) ** 2
        fraction = potential / de if de > 0 else 0.0
        distance_ratio = r / r0 if r0 > 0 else math.inf

        energy_trigger = fraction >= break_frac
        distance_trigger = distance_ratio >= break_distance_factor
        if energy_trigger or distance_trigger:
            if energy_trigger and distance_trigger:
                reason = "energy_and_distance"
            elif energy_trigger:
                reason = "energy"
            else:
                reason = "distance"
            broken.append(
                {
                    "bond_index": idx,
                    "i": i,
                    "j": j,
                    "distance": float(r),
                    "r0": float(r0),
                    "distance_ratio": float(distance_ratio),
                    "V": float(potential),
                    "De": float(de),
                    "fraction": float(fraction),
                    "reason": reason,
                }
            )
    return broken
