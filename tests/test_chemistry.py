from __future__ import annotations

import pytest

from molecule_lab.api.schemas import GraphAtom, GraphBond, MoleculeGraph
from molecule_lab.chem import calculate_static_info, graph_to_smiles, validate_graph
from molecule_lab.core.errors import MoleculeValidationError


def test_graph_to_smiles_for_single_carbon() -> None:
    graph = MoleculeGraph(atoms=[GraphAtom(id="c1", symbol="C", x=0, y=0)])

    assert graph_to_smiles(graph) == "C"


def test_static_properties_for_methane_smiles() -> None:
    info = calculate_static_info("C")

    assert info.formula == "CH4"
    assert info.properties.num_atoms == 1
    assert info.properties.num_bonds == 0
    assert info.properties.is_aromatic is False


def test_validation_rejects_duplicate_atom_ids() -> None:
    graph = MoleculeGraph(
        atoms=[
            GraphAtom(id="a", symbol="C", x=0, y=0),
            GraphAtom(id="a", symbol="O", x=1, y=0),
        ]
    )

    with pytest.raises(MoleculeValidationError, match="únicos"):
        validate_graph(graph)


def test_validation_rejects_disconnected_molecule() -> None:
    graph = MoleculeGraph(
        atoms=[
            GraphAtom(id="c1", symbol="C", x=0, y=0),
            GraphAtom(id="c2", symbol="C", x=1, y=0),
        ],
        bonds=[],
    )

    with pytest.raises(MoleculeValidationError, match="conectado"):
        validate_graph(graph)


def test_graph_with_bond_converts_to_ethane_smiles() -> None:
    graph = MoleculeGraph(
        atoms=[
            GraphAtom(id="c1", symbol="C", x=0, y=0),
            GraphAtom(id="c2", symbol="C", x=1, y=0),
        ],
        bonds=[GraphBond(from_="c1", to="c2", order=1)],
    )

    assert graph_to_smiles(graph) == "CC"
