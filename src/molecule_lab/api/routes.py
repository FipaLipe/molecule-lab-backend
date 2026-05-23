"""FastAPI routes for creating and streaming simulations."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from molecule_lab.api.registry import SimulationRecord, SimulationRegistry
from molecule_lab.api.schemas import (
    HealthResponse,
    SimulationCreateRequest,
    SimulationCreateResponse,
)
from molecule_lab.chem import calculate_static_info, graph_to_smiles
from molecule_lab.core.config import settings
from molecule_lab.core.errors import (
    MoleculeValidationError,
    SimulationError,
    SimulationNotFoundError,
)
from molecule_lab.simulation import get_preset, run_simulation

log = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_prefix)
registry = SimulationRegistry()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name)


@router.post("/simulations", response_model=SimulationCreateResponse, status_code=201)
def create_simulation(req: SimulationCreateRequest) -> SimulationCreateResponse:
    try:
        get_preset(req.preset)
        smiles = graph_to_smiles(req.graph)
        molecule = calculate_static_info(smiles)
    except MoleculeValidationError as exc:
        raise HTTPException(status_code=422, detail=_error("invalid_molecule", str(exc)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_error("invalid_preset", str(exc)))

    record = registry.create(
        smiles=molecule.smiles,
        preset=req.preset,
        seed=req.seed,
        molecule=molecule,
    )
    return SimulationCreateResponse(
        simulation_id=record.simulation_id,
        preset=req.preset,
        molecule=molecule,
        events_url=f"{settings.api_prefix}/simulations/{record.simulation_id}/events",
    )


@router.get("/simulations/{simulation_id}/events")
def stream_simulation_events(simulation_id: str) -> StreamingResponse:
    try:
        record = registry.get(simulation_id)
    except SimulationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_error("simulation_not_found", str(exc)))

    return StreamingResponse(
        _event_stream(record),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _event_stream(record: SimulationRecord):
    yield _sse(
        "metadata",
        {
            "simulation_id": record.simulation_id,
            "preset": record.preset,
            "seed": record.seed,
            "molecule": record.molecule.model_dump(),
        },
    )
    try:
        for event in run_simulation(
            record.smiles,
            preset_name=record.preset,
            seed=record.seed,
        ):
            yield _sse(event.event, event.payload)
    except SimulationError as exc:
        log.exception("Simulation failed")
        yield _sse("error", _error("simulation_error", str(exc)))
    except Exception as exc:
        log.exception("Unexpected simulation failure")
        yield _sse("error", _error("simulation_error", f"Erro inesperado: {exc}"))


def _sse(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return f"event: {event}\ndata: {data}\n\n"


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}
