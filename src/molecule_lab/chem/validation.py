"""Validation for user-drawn molecule graphs before RDKit conversion."""

from __future__ import annotations

from collections import deque

from molecule_lab.api.schemas import MoleculeGraph
from molecule_lab.core.config import settings
from molecule_lab.core.errors import MoleculeValidationError


def validate_graph(graph: MoleculeGraph) -> None:
    if len(graph.atoms) > settings.max_atoms:
        raise MoleculeValidationError(
            f"Molécula excede o limite de {settings.max_atoms} átomos."
        )
    if len(graph.bonds) > settings.max_bonds:
        raise MoleculeValidationError(
            f"Molécula excede o limite de {settings.max_bonds} ligações."
        )

    ids = [atom.id for atom in graph.atoms]
    if len(ids) != len(set(ids)):
        raise MoleculeValidationError("IDs de átomos devem ser únicos.")

    atom_ids = set(ids)
    for atom in graph.atoms:
        if atom.symbol not in settings.allowed_symbols:
            allowed = ", ".join(sorted(settings.allowed_symbols))
            raise MoleculeValidationError(
                f"Elemento '{atom.symbol}' não é suportado. Use: {allowed}."
            )

    seen_bonds: set[tuple[str, str]] = set()
    adjacency = {atom_id: set() for atom_id in atom_ids}
    for bond in graph.bonds:
        if bond.from_ not in atom_ids or bond.to not in atom_ids:
            raise MoleculeValidationError("Ligação referencia átomo inexistente.")
        if bond.from_ == bond.to:
            raise MoleculeValidationError("Ligação não pode conectar o átomo a ele mesmo.")

        pair = tuple(sorted((bond.from_, bond.to)))
        if pair in seen_bonds:
            raise MoleculeValidationError("Ligações duplicadas não são permitidas.")
        seen_bonds.add(pair)
        adjacency[bond.from_].add(bond.to)
        adjacency[bond.to].add(bond.from_)

    if len(atom_ids) > 1 and not _is_connected(adjacency):
        raise MoleculeValidationError("A molécula deve ser um único componente conectado.")


def _is_connected(adjacency: dict[str, set[str]]) -> bool:
    first = next(iter(adjacency))
    visited = {first}
    queue: deque[str] = deque([first])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == len(adjacency)
