"""Application settings with conservative defaults for the classroom demo."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    app_name: str = "Molecule Lab API"
    api_prefix: str = "/api"
    allowed_origins: tuple[str, ...] = ("*",)
    max_atoms: int = 60
    max_bonds: int = 90
    allowed_symbols: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
        )
    )
    simulation_cache_size: int = 128


settings = Settings()
