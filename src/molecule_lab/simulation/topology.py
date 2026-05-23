"""RDKit structure building and topology extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molecule_lab.core.errors import SimulationError
from molecule_lab.simulation.parameters import (
    ANGLE_K_HYB,
    DIHEDRAL_PARAMS,
    EPSILON_MAP,
    SIGMA_SCALE,
    bond_morse_parameters,
)


@dataclass
class Topology:
    pos: np.ndarray
    masses: np.ndarray
    radii: np.ndarray
    symbols: list[str]
    charges: np.ndarray
    bonds: list[tuple[int, int, float, float, float]]
    bonded_pairs: set[tuple[int, int]]
    one_three: set[tuple[int, int]]
    one_four: set[tuple[int, int]]
    angles: list[tuple[int, int, int, float, float]]
    dihedrals: list[tuple[int, int, int, int, float, float, float]]
    shake_bonds: list[tuple[int, int, float]]
    bond_i: np.ndarray
    bond_j: np.ndarray
    bond_r0: np.ndarray
    bond_de: np.ndarray
    bond_alpha: np.ndarray
    angle_i: np.ndarray
    angle_c: np.ndarray
    angle_j: np.ndarray
    angle_theta0: np.ndarray
    angle_k: np.ndarray
    dihedral_i: np.ndarray
    dihedral_j: np.ndarray
    dihedral_k: np.ndarray
    dihedral_l: np.ndarray
    dihedral_v1: np.ndarray
    dihedral_v2: np.ndarray
    dihedral_v3: np.ndarray


def build_molecule(smiles: str, seed: int = 0xF00D) -> Chem.Mol:
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise SimulationError("SMILES inválido.")

    mol = Chem.AddHs(base)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise SimulationError("ETKDGv3 falhou ao gerar a geometria inicial.")

    status = AllChem.UFFOptimizeMolecule(mol, maxIters=2_000)
    if status < 0:
        raise SimulationError("UFF falhou ao otimizar a geometria inicial.")
    return mol


def get_partial_charges(mol: Chem.Mol) -> np.ndarray:
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for i in range(mol.GetNumAtoms()):
        q = mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge")
        charges.append(q if math.isfinite(q) else 0.0)
    return np.array(charges, dtype=float)


def lj_params(si: str, sj: str, ri: float, rj: float) -> tuple[float, float]:
    sigma = SIGMA_SCALE * (ri + rj)
    epsilon = math.sqrt(EPSILON_MAP.get(si, 0.08) * EPSILON_MAP.get(sj, 0.08))
    return sigma, epsilon


def build_topology(mol: Chem.Mol) -> Topology:
    conf = mol.GetConformer()
    pt = Chem.GetPeriodicTable()
    atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
    masses = np.array([atom.GetMass() for atom in atoms])
    radii = np.array([pt.GetRcovalent(atom.GetAtomicNum()) for atom in atoms])
    symbols = [atom.GetSymbol() for atom in atoms]
    hybs = [atom.GetHybridization() for atom in atoms]
    charges = get_partial_charges(mol)

    pos = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )

    bonds: list[tuple[int, int, float, float, float]] = []
    neighbors = {i: set() for i in range(mol.GetNumAtoms())}
    bond_bt_map = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r0 = float(np.linalg.norm(pos[j] - pos[i]))
        de, kh = bond_morse_parameters(symbols[i], symbols[j], bond.GetBondType())
        alpha = math.sqrt(max(kh, 1e-12) / (2.0 * max(de, 1e-12)))
        bonds.append((i, j, r0, de, alpha))
        neighbors[i].add(j)
        neighbors[j].add(i)
        bond_bt_map[(min(i, j), max(i, j))] = bond.GetBondType()

    bonded_pairs = {(min(i, j), max(i, j)) for i, j, *_ in bonds}

    one_three: set[tuple[int, int]] = set()
    angles: list[tuple[int, int, int, float, float]] = []
    for c in range(mol.GetNumAtoms()):
        nbrs = sorted(neighbors[c])
        k_theta = ANGLE_K_HYB.get(hybs[c], 100.0)
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                ii, jj = nbrs[a], nbrs[b]
                one_three.add((min(ii, jj), max(ii, jj)))
                rci = pos[ii] - pos[c]
                rcj = pos[jj] - pos[c]
                r_ci = np.linalg.norm(rci) + 1e-12
                r_cj = np.linalg.norm(rcj) + 1e-12
                cos_t = np.clip(np.dot(rci, rcj) / (r_ci * r_cj), -1.0, 1.0)
                angles.append((ii, c, jj, math.acos(cos_t), k_theta))

    dihedrals: list[tuple[int, int, int, int, float, float, float]] = []
    seen_dih = set()
    for (p, q), bt in bond_bt_map.items():
        v1, v2, v3 = DIHEDRAL_PARAMS.get(bt, (0.0, 0.0, 1.5))
        if v1 == v2 == v3 == 0.0:
            continue
        for i in neighbors[p]:
            if i == q:
                continue
            for l in neighbors[q]:
                if l == p or l == i:
                    continue
                tup = (i, p, q, l)
                key = min(tup, (l, q, p, i))
                if key in seen_dih:
                    continue
                seen_dih.add(key)
                dihedrals.append((i, p, q, l, v1, v2, v3))

    one_four: set[tuple[int, int]] = set()
    for i, _j, _k, l, *_ in dihedrals:
        pair = (min(i, l), max(i, l))
        if pair not in bonded_pairs and pair not in one_three:
            one_four.add(pair)

    # SHAKE is kept available for non-reactive runs, but presets now disable it
    # by default. If enabled, X-H bonds cannot break.
    shake_bonds = [
        (i, j, r0)
        for i, j, r0, _de, _alpha in bonds
        if symbols[i] == "H" or symbols[j] == "H"
    ]
    bond_arrays = _bond_arrays(bonds)
    angle_arrays = _angle_arrays(angles)
    dihedral_arrays = _dihedral_arrays(dihedrals)

    return Topology(
        pos=pos,
        masses=masses,
        radii=radii,
        symbols=symbols,
        charges=charges,
        bonds=bonds,
        bonded_pairs=bonded_pairs,
        one_three=one_three,
        one_four=one_four,
        angles=angles,
        dihedrals=dihedrals,
        shake_bonds=shake_bonds,
        bond_i=bond_arrays[0],
        bond_j=bond_arrays[1],
        bond_r0=bond_arrays[2],
        bond_de=bond_arrays[3],
        bond_alpha=bond_arrays[4],
        angle_i=angle_arrays[0],
        angle_c=angle_arrays[1],
        angle_j=angle_arrays[2],
        angle_theta0=angle_arrays[3],
        angle_k=angle_arrays[4],
        dihedral_i=dihedral_arrays[0],
        dihedral_j=dihedral_arrays[1],
        dihedral_k=dihedral_arrays[2],
        dihedral_l=dihedral_arrays[3],
        dihedral_v1=dihedral_arrays[4],
        dihedral_v2=dihedral_arrays[5],
        dihedral_v3=dihedral_arrays[6],
    )


def _bond_arrays(
    bonds: list[tuple[int, int, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not bonds:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )
    return (
        np.array([item[0] for item in bonds], dtype=np.int32),
        np.array([item[1] for item in bonds], dtype=np.int32),
        np.array([item[2] for item in bonds], dtype=float),
        np.array([item[3] for item in bonds], dtype=float),
        np.array([item[4] for item in bonds], dtype=float),
    )


def _angle_arrays(
    angles: list[tuple[int, int, int, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not angles:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0),
            np.empty(0),
        )
    return (
        np.array([item[0] for item in angles], dtype=np.int32),
        np.array([item[1] for item in angles], dtype=np.int32),
        np.array([item[2] for item in angles], dtype=np.int32),
        np.array([item[3] for item in angles], dtype=float),
        np.array([item[4] for item in angles], dtype=float),
    )


def _dihedral_arrays(
    dihedrals: list[tuple[int, int, int, int, float, float, float]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not dihedrals:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )
    return (
        np.array([item[0] for item in dihedrals], dtype=np.int32),
        np.array([item[1] for item in dihedrals], dtype=np.int32),
        np.array([item[2] for item in dihedrals], dtype=np.int32),
        np.array([item[3] for item in dihedrals], dtype=np.int32),
        np.array([item[4] for item in dihedrals], dtype=float),
        np.array([item[5] for item in dihedrals], dtype=float),
        np.array([item[6] for item in dihedrals], dtype=float),
    )
