"""Conversion between frontend graph data and RDKit molecules."""

from __future__ import annotations

from rdkit import Chem

from molecule_lab.api.schemas import MoleculeGraph
from molecule_lab.chem.validation import validate_graph
from molecule_lab.core.errors import MoleculeValidationError


_BOND_TYPE_FROM_ORDER = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def graph_to_rdkit_mol(graph: MoleculeGraph) -> Chem.Mol:
    validate_graph(graph)

    mol = Chem.RWMol()
    id_to_idx: dict[str, int] = {}
    try:
        for atom in graph.atoms:
            rd_atom = Chem.Atom(atom.symbol)
            idx = mol.AddAtom(rd_atom)
            id_to_idx[atom.id] = idx

        for bond in graph.bonds:
            mol.AddBond(
                id_to_idx[bond.from_],
                id_to_idx[bond.to],
                _BOND_TYPE_FROM_ORDER[bond.order],
            )

        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise MoleculeValidationError(f"Molécula inválida para o RDKit: {exc}") from exc

    return mol.GetMol()


def graph_to_smiles(graph: MoleculeGraph) -> str:
    mol = graph_to_rdkit_mol(graph)
    return Chem.MolToSmiles(mol, canonical=True)
