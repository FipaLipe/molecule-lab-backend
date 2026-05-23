"""BAOAB Langevin integration and momentum constraints."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from molecule_lab.simulation.parameters import (
    KCAL_MOL_A_TO_AMU_A_FS2,
    K_B_KCAL_PER_MOL_K,
    SimulationPreset,
)


ForceFn = Callable[[np.ndarray], tuple[np.ndarray, float, np.ndarray]]


def shake_positions(
    pos: np.ndarray,
    masses: np.ndarray,
    shake_bonds: list[tuple[int, int, float]],
    preset: SimulationPreset,
) -> np.ndarray:
    for _ in range(preset.shake_max_iter):
        converged = True
        for i, j, d0 in shake_bonds:
            rij = pos[j] - pos[i]
            d2 = float(np.dot(rij, rij))
            d0_2 = d0 * d0
            if abs(d2 - d0_2) < preset.shake_tol * d0_2:
                continue
            converged = False
            lam = (d2 - d0_2) / (2.0 * (1 / masses[i] + 1 / masses[j]) * d2)
            pos[i] += lam / masses[i] * rij
            pos[j] -= lam / masses[j] * rij
        if converged:
            break
    return pos


def rattle_velocities(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    shake_bonds: list[tuple[int, int, float]],
    preset: SimulationPreset,
) -> np.ndarray:
    for _ in range(preset.shake_max_iter):
        converged = True
        for i, j, d0 in shake_bonds:
            rij = pos[j] - pos[i]
            vij = vel[j] - vel[i]
            rv = float(np.dot(rij, vij))
            if abs(rv) < preset.shake_tol:
                continue
            converged = False
            mu = rv / ((1 / masses[i] + 1 / masses[j]) * d0 * d0)
            vel[i] += mu / masses[i] * rij
            vel[j] -= mu / masses[j] * rij
        if converged:
            break
    return vel


def baoab_step(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    forces: np.ndarray,
    target_temperature: float,
    rng: np.random.Generator,
    shake_bonds: list[tuple[int, int, float]],
    force_fn: ForceFn,
    preset: SimulationPreset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    # forces are kcal/mol/Å; masses are amu; acceleration is Å/fs².
    acc = KCAL_MOL_A_TO_AMU_A_FS2 * forces / masses[:, None]
    vel = vel + 0.5 * preset.dt * acc
    pos = pos + 0.5 * preset.dt * vel
    if shake_bonds:
        pos = shake_positions(pos, masses, shake_bonds, preset)

    c = math.exp(-preset.gamma * preset.dt)
    # Maxwell-Boltzmann variance for velocities in Å/fs.
    noise_std = np.sqrt(
        max(0.0, 1 - c * c)
        * K_B_KCAL_PER_MOL_K
        * target_temperature
        * KCAL_MOL_A_TO_AMU_A_FS2
        / masses
    )[:, None]
    vel = c * vel + noise_std * rng.standard_normal(vel.shape)

    pos = pos + 0.5 * preset.dt * vel
    if shake_bonds:
        pos = shake_positions(pos, masses, shake_bonds, preset)

    new_forces, potential, atom_energies = force_fn(pos)
    new_acc = KCAL_MOL_A_TO_AMU_A_FS2 * new_forces / masses[:, None]
    vel = vel + 0.5 * preset.dt * new_acc
    if shake_bonds:
        vel = rattle_velocities(pos, vel, masses, shake_bonds, preset)
    return pos, vel, new_forces, potential, atom_energies


def initialize_velocities(masses: np.ndarray, temperature: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vel = rng.normal(size=(len(masses), 3)) * np.sqrt(
        K_B_KCAL_PER_MOL_K
        * max(temperature, 1e-6)
        * KCAL_MOL_A_TO_AMU_A_FS2
        / masses
    )[:, None]
    vel -= np.average(vel, axis=0, weights=masses)
    return vel


def kinetic_temperature(
    vel: np.ndarray, masses: np.ndarray, degrees_of_freedom: int | None = None
) -> tuple[float, float, int]:
    # Convert amu·Å²/fs² to kcal/mol.
    kinetic = 0.5 * np.sum(masses[:, None] * vel**2) / KCAL_MOL_A_TO_AMU_A_FS2
    n_f = max(1, 3 * len(masses) - 6) if degrees_of_freedom is None else degrees_of_freedom
    return 2.0 * kinetic / (n_f * K_B_KCAL_PER_MOL_K), float(kinetic), n_f


def temperature_ramp(step: int, n_steps: int, t0: float, t1: float) -> float:
    return t0 + (t1 - t0) * step / max(n_steps - 1, 1)


def remove_linear_momentum(vel: np.ndarray, masses: np.ndarray) -> np.ndarray:
    vel -= np.average(vel, axis=0, weights=masses)
    return vel


def remove_angular_momentum(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray) -> np.ndarray:
    total_mass = masses.sum()
    r_com = (masses[:, None] * pos).sum(0) / total_mass
    r_rel = pos - r_com
    angular_momentum = (masses[:, None] * np.cross(r_rel, vel)).sum(0)
    inertia = np.zeros((3, 3))
    for k in range(len(masses)):
        r = r_rel[k]
        inertia += masses[k] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    try:
        omega = np.linalg.solve(inertia, angular_momentum)
        vel -= np.cross(omega[None, :], r_rel)
    except np.linalg.LinAlgError:
        pass
    return vel
