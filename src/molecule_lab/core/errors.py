"""Domain exceptions converted to HTTP errors by the API layer."""

from __future__ import annotations


class MoleculeLabError(Exception):
    """Base class for expected application errors."""


class MoleculeValidationError(MoleculeLabError):
    """Raised when a user-provided molecular graph is invalid."""


class SimulationNotFoundError(MoleculeLabError):
    """Raised when a simulation id does not exist in the registry."""


class SimulationError(MoleculeLabError):
    """Raised when the molecular dynamics simulation cannot run."""
