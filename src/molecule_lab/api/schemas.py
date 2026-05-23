"""Pydantic schemas exposed by the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


PresetName = Literal["fast", "balanced", "debug"]


class GraphAtom(BaseModel):
    id: str = Field(..., min_length=1, max_length=80)
    symbol: str = Field(..., min_length=1, max_length=2)
    x: float
    y: float

    @field_validator("id", "symbol")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class GraphBond(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(..., alias="from", min_length=1, max_length=80)
    to: str = Field(..., min_length=1, max_length=80)
    order: int = Field(default=1, ge=1, le=3)

    @field_validator("from_", "to")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class MoleculeGraph(BaseModel):
    atoms: list[GraphAtom] = Field(..., min_length=1)
    bonds: list[GraphBond] = Field(default_factory=list)


class SimulationCreateRequest(BaseModel):
    graph: MoleculeGraph
    preset: PresetName = "fast"
    seed: int | None = None


class MoleculeProperties(BaseModel):
    num_atoms: int
    num_bonds: int
    num_rings: int
    is_aromatic: bool


class StaticMoleculeInfo(BaseModel):
    smiles: str
    formula: str
    molecular_weight: float
    properties: MoleculeProperties


class SimulationCreateResponse(BaseModel):
    simulation_id: str
    preset: PresetName
    molecule: StaticMoleculeInfo
    events_url: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class ApiError(BaseModel):
    code: str
    message: str
