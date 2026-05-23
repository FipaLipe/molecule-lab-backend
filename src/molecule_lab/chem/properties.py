"""Static molecular descriptors computed with RDKit."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from molecule_lab.api.schemas import MoleculeProperties, StaticMoleculeInfo
from molecule_lab.core.errors import MoleculeValidationError


def calculate_static_info(smiles: str) -> StaticMoleculeInfo:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeValidationError("SMILES inválido.")

    return StaticMoleculeInfo(
        smiles=Chem.MolToSmiles(mol, canonical=True),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        properties=MoleculeProperties(
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            num_rings=mol.GetRingInfo().NumRings(),
            is_aromatic=any(atom.GetIsAromatic() for atom in mol.GetAtoms()),
        ),
    )
