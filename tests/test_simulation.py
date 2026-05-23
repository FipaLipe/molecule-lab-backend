from __future__ import annotations

import numpy as np

from molecule_lab.simulation import run_simulation
from molecule_lab.simulation.detection import detect_broken_bonds
from molecule_lab.simulation.engine import _RESULT_CACHE
from molecule_lab.simulation.parameters import get_preset
from molecule_lab.simulation.topology import build_molecule, build_topology


def test_presets_are_available() -> None:
    assert get_preset("fast").n_steps > get_preset("debug").n_steps
    assert get_preset("balanced").event_every > 0


def test_topology_builds_hydrogenated_methane() -> None:
    mol = build_molecule("C", seed=7)
    topology = build_topology(mol)

    assert len(topology.symbols) == 5
    assert topology.symbols.count("H") == 4
    assert len(topology.bonds) == 4
    assert topology.pos.shape == (5, 3)


def test_detect_broken_bond_controlled_case() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    bonds = [(0, 1, 1.0, 150.0, 1.0)]

    broken = detect_broken_bonds(pos, bonds, break_frac=0.95)

    assert len(broken) == 1
    assert broken[0]["bond_index"] == 0


def test_debug_simulation_is_deterministic_and_cached() -> None:
    _RESULT_CACHE.clear()

    first = list(run_simulation("C", preset_name="debug", seed=123))
    second = list(run_simulation("C", preset_name="debug", seed=123))

    first_result = next(event.payload for event in first if event.event == "result")
    second_result = next(event.payload for event in second if event.event == "result")
    assert first_result["result"] == "stable"
    assert second[0].event == "cache_hit"
    assert second_result["result"] == first_result["result"]
