"""In-memory simulation registry used by the SSE API."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from threading import Lock

from molecule_lab.api.schemas import StaticMoleculeInfo
from molecule_lab.core.errors import SimulationNotFoundError


@dataclass(frozen=True)
class SimulationRecord:
    simulation_id: str
    smiles: str
    preset: str
    seed: int | None
    molecule: StaticMoleculeInfo
    created_at: float


class SimulationRegistry:
    def __init__(self) -> None:
        self._records: dict[str, SimulationRecord] = {}
        self._lock = Lock()

    def create(
        self,
        smiles: str,
        preset: str,
        seed: int | None,
        molecule: StaticMoleculeInfo,
    ) -> SimulationRecord:
        simulation_id = uuid.uuid4().hex
        record = SimulationRecord(
            simulation_id=simulation_id,
            smiles=smiles,
            preset=preset,
            seed=seed,
            molecule=molecule,
            created_at=time.time(),
        )
        with self._lock:
            self._records[simulation_id] = record
        return record

    def get(self, simulation_id: str) -> SimulationRecord:
        with self._lock:
            record = self._records.get(simulation_id)
        if record is None:
            raise SimulationNotFoundError("Simulação não encontrada.")
        return record
