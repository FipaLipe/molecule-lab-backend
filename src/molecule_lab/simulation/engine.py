"""High-level molecular dynamics runner and progress event generation."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from molecule_lab.core.config import settings
from molecule_lab.core.errors import SimulationError
from molecule_lab.simulation.detection import detect_broken_bonds
from molecule_lab.simulation.forces import VerletList, forces_and_energy
from molecule_lab.simulation.integrator import (
    baoab_step,
    initialize_velocities,
    kinetic_temperature,
    remove_angular_momentum,
    remove_linear_momentum,
    temperature_ramp,
)
from molecule_lab.simulation.parameters import SimulationPreset, get_preset
from molecule_lab.simulation.topology import build_molecule, build_topology


ResultKind = Literal["stable", "break"]
EventKind = Literal["progress", "result", "cache_hit"]


@dataclass(frozen=True)
class SimulationEvent:
    event: EventKind
    payload: dict[str, Any]


_RESULT_CACHE: OrderedDict[tuple[str, str, int], dict[str, Any]] = OrderedDict()


def run_simulation(
    smiles: str, preset_name: str = "fast", seed: int | None = None
) -> Iterator[SimulationEvent]:
    preset = get_preset(preset_name)
    run_seed = preset.seed if seed is None else seed
    cache_key = (smiles, preset.name, run_seed)
    cached = _cache_get(cache_key)
    if cached is not None:
        yield SimulationEvent(
            "cache_hit",
            {
                "smiles": smiles,
                "preset": preset.name,
                "seed": run_seed,
                "message": "Resultado recuperado do cache em memória.",
            },
        )
        yield SimulationEvent("result", {**cached, "cached": True})
        return

    final_payload: dict[str, Any] | None = None
    for event in _run_uncached(smiles, preset, run_seed):
        if event.event == "result":
            final_payload = event.payload
        yield event
    if final_payload is not None:
        _cache_put(cache_key, final_payload)


def _run_uncached(
    smiles: str, preset: SimulationPreset, seed: int
) -> Iterator[SimulationEvent]:
    start = time.perf_counter()
    try:
        mol = build_molecule(smiles, seed=seed)
        topology = build_topology(mol)
    except Exception as exc:
        if isinstance(exc, SimulationError):
            raise
        raise SimulationError(str(exc)) from exc

    pos = topology.pos.copy()
    active_shake_bonds = topology.shake_bonds if preset.use_shake else []
    vlist = VerletList(preset.r_cut, preset.r_skin)
    vlist.build(pos, topology, preset)

    vel = initialize_velocities(topology.masses, preset.temperature_start, seed=seed)
    vel = remove_angular_momentum(pos, vel, topology.masses)
    forces, potential, atom_energy = forces_and_energy(pos, topology, vlist, preset)
    rng = np.random.default_rng(seed)

    def force_fn(current_pos: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        if vlist.needs_rebuild(current_pos):
            vlist.build(current_pos, topology, preset)
        return forces_and_energy(current_pos, topology, vlist, preset)

    temperatures: list[float] = []
    target_temperatures: list[float] = []
    persist_count = {idx: 0 for idx in range(len(topology.bonds))}
    for step in range(preset.n_steps):
        target_temperature = temperature_ramp(
            step,
            preset.n_steps,
            preset.temperature_start,
            preset.temperature_end,
        )
        pos, vel, forces, potential, atom_energy = baoab_step(
            pos,
            vel,
            topology.masses,
            forces,
            target_temperature,
            rng,
            active_shake_bonds,
            force_fn,
            preset,
        )
        vel = remove_linear_momentum(vel, topology.masses)
        if preset.reset_ang_mom > 0 and (step + 1) % preset.reset_ang_mom == 0:
            vel = remove_angular_momentum(pos, vel, topology.masses)

        current_temperature, kinetic, _ = kinetic_temperature(vel, topology.masses)
        temperatures.append(float(current_temperature))
        target_temperatures.append(float(target_temperature))

        broken_now = detect_broken_bonds(
            pos,
            topology.bonds,
            preset.break_frac,
            preset.break_distance_factor,
        )
        active = {int(entry["bond_index"]) for entry in broken_now}
        for idx in persist_count:
            persist_count[idx] = persist_count[idx] + 1 if idx in active else 0
        persistent = [
            entry
            for entry in broken_now
            if persist_count[int(entry["bond_index"])] >= preset.break_persistence
        ]

        if step % preset.event_every == 0 or step == preset.n_steps - 1:
            yield SimulationEvent(
                "progress",
                {
                    "step": step,
                    "n_steps": preset.n_steps,
                    "progress": round((step + 1) / preset.n_steps, 4),
                    "target_temperature": float(target_temperature),
                    "current_temperature": float(current_temperature),
                    "potential_energy": float(potential),
                    "kinetic_energy": float(kinetic),
                    "candidate_broken_bonds": _format_broken_bonds(
                        broken_now, topology.symbols
                    ),
                },
            )

        if persistent:
            yield SimulationEvent(
                "result",
                _result_payload(
                    "break",
                    step,
                    float(target_temperature),
                    float(current_temperature),
                    persistent,
                    topology.symbols,
                    temperatures,
                    target_temperatures,
                    start,
                ),
            )
            return

    yield SimulationEvent(
        "result",
        _result_payload(
            "stable",
            None,
            None,
            None,
            [],
            topology.symbols,
            temperatures,
            target_temperatures,
            start,
        ),
    )


def _result_payload(
    result: ResultKind,
    break_step: int | None,
    break_temperature: float | None,
    current_break_temperature: float | None,
    broken_bonds: list[dict[str, float | int | str]],
    symbols: list[str],
    temperatures: list[float],
    target_temperatures: list[float],
    start_time: float,
) -> dict[str, Any]:
    return {
        "result": result,
        "break_step": break_step,
        "break_temperature": break_temperature,
        "current_break_temperature": current_break_temperature,
        "broken_bonds": _format_broken_bonds(broken_bonds, symbols),
        "symbols": symbols,
        "temperatures": temperatures,
        "target_temperatures": target_temperatures,
        "elapsed_seconds": round(time.perf_counter() - start_time, 4),
        "cached": False,
    }


def _format_broken_bonds(
    broken_bonds: list[dict[str, float | int | str]], symbols: list[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for bond in broken_bonds:
        i = int(bond["i"])
        j = int(bond["j"])
        output.append(
            {
                "atom_i": symbols[i],
                "atom_j": symbols[j],
                "atom_i_index": i,
                "atom_j_index": j,
                "distance": round(float(bond["distance"]), 3),
                "r0": round(float(bond["r0"]), 3),
                "distance_ratio": round(float(bond.get("distance_ratio", 0.0)), 3),
                "fraction": round(float(bond["V"]) / float(bond["De"]), 3),
                "De": round(float(bond["De"]), 3),
                "reason": str(bond.get("reason", "unknown")),
            }
        )
    return output


def _cache_get(cache_key: tuple[str, str, int]) -> dict[str, Any] | None:
    value = _RESULT_CACHE.get(cache_key)
    if value is None:
        return None
    _RESULT_CACHE.move_to_end(cache_key)
    return value


def _cache_put(cache_key: tuple[str, str, int], value: dict[str, Any]) -> None:
    _RESULT_CACHE[cache_key] = value
    _RESULT_CACHE.move_to_end(cache_key)
    while len(_RESULT_CACHE) > settings.simulation_cache_size:
        _RESULT_CACHE.popitem(last=False)
