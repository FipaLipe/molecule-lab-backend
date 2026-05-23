"""Simulation constants and presets.

The engine uses a lightweight, educational reactive-MD approximation.
Units used throughout the simulation layer:

- distance: Å
- time: fs
- mass: atomic mass unit (amu)
- energy: kcal/mol
- temperature: K

The parameters below are still simplified. They are meant to give physically
reasonable trends and temperatures that can be calibrated against literature,
not ab-initio or force-field-grade predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

# Thermodynamic / unit-conversion constants for the unit system above.
K_B_KCAL_PER_MOL_K = 0.00198720425864083
KCAL_MOL_A_TO_AMU_A_FS2 = 4.184e-4
COULOMB_KCAL_A_PER_MOL_E2 = 332.063713299


@dataclass(frozen=True)
class SimulationPreset:
    name: str
    temperature_start: float
    temperature_end: float
    n_steps: int
    dt: float
    gamma: float
    reset_ang_mom: int
    break_frac: float
    break_distance_factor: float
    break_persistence: int
    r_cut: float
    r_skin: float
    k_coulomb: float
    lj_scale_14: float
    use_shake: bool
    shake_tol: float
    shake_max_iter: int
    save_every: int
    event_every: int
    seed: int


# The Coulomb term is deliberately screened by an effective dielectric of 4.
# Vacuum Coulomb would be COULOMB_KCAL_A_PER_MOL_E2 and is usually too harsh
# for this small educational model with Gasteiger charges.
_SCREENED_COULOMB = COULOMB_KCAL_A_PER_MOL_E2 / 4.0

PRESETS: dict[str, SimulationPreset] = {
    "fast": SimulationPreset(
        name="fast",
        temperature_start=298.15,
        temperature_end=10_000.0,
        n_steps=5_000,
        dt=0.10,  # fs
        gamma=0.05,  # fs^-1, strong thermostat for a fast heating ramp
        reset_ang_mom=200,
        break_frac=0.50,
        break_distance_factor=1.60,
        break_persistence=8,
        r_cut=8.0,
        r_skin=1.5,
        k_coulomb=_SCREENED_COULOMB,
        lj_scale_14=0.5,
        use_shake=False,
        shake_tol=1e-8,
        shake_max_iter=50,
        save_every=5,
        event_every=10,
        seed=42,
    ),
    "balanced": SimulationPreset(
        name="balanced",
        temperature_start=298.15,
        temperature_end=10_000.0,
        n_steps=20_000,
        dt=0.05,  # fs
        gamma=0.05,  # fs^-1
        reset_ang_mom=400,
        break_frac=0.50,
        break_distance_factor=1.60,
        break_persistence=16,
        r_cut=8.0,
        r_skin=1.5,
        k_coulomb=_SCREENED_COULOMB,
        lj_scale_14=0.5,
        use_shake=False,
        shake_tol=1e-8,
        shake_max_iter=50,
        save_every=5,
        event_every=20,
        seed=42,
    ),
    "debug": SimulationPreset(
        name="debug",
        temperature_start=298.15,
        temperature_end=2_000.0,
        n_steps=500,
        dt=0.10,  # fs
        gamma=0.05,  # fs^-1
        reset_ang_mom=50,
        break_frac=0.50,
        break_distance_factor=1.60,
        break_persistence=4,
        r_cut=8.0,
        r_skin=1.5,
        k_coulomb=_SCREENED_COULOMB,
        lj_scale_14=0.5,
        use_shake=False,
        shake_tol=1e-8,
        shake_max_iter=50,
        save_every=1,
        event_every=20,
        seed=7,
    ),
}


# Fallback Morse parameters by formal bond order: (D_e, k_harmonic)
# D_e in kcal/mol and k in kcal/mol/Å².
BOND_ORDER_DEFAULTS: dict[Chem.rdchem.BondType, tuple[float, float]] = {
    Chem.rdchem.BondType.SINGLE: (85.0, 320.0),
    Chem.rdchem.BondType.DOUBLE: (145.0, 570.0),
    Chem.rdchem.BondType.TRIPLE: (200.0, 900.0),
    Chem.rdchem.BondType.AROMATIC: (120.0, 470.0),
}

# Pair-specific Morse parameters: sorted atom pair + bond label -> (D_e, k).
# These are representative gas-phase bond-dissociation / stretching values,
# intentionally rounded for an educational model.
BOND_PAIR_PARAMS: dict[tuple[str, str, str], tuple[float, float]] = {
    ("C", "H", "SINGLE"): (105.0, 340.0),
    ("H", "O", "SINGLE"): (110.0, 450.0),
    ("H", "N", "SINGLE"): (93.0, 430.0),
    ("C", "C", "SINGLE"): (83.0, 310.0),
    ("C", "C", "DOUBLE"): (146.0, 570.0),
    ("C", "C", "TRIPLE"): (200.0, 960.0),
    ("C", "C", "AROMATIC"): (120.0, 470.0),
    ("C", "O", "SINGLE"): (86.0, 320.0),
    ("C", "O", "DOUBLE"): (179.0, 700.0),
    ("C", "N", "SINGLE"): (73.0, 337.0),
    ("C", "N", "DOUBLE"): (147.0, 650.0),
    ("C", "N", "TRIPLE"): (213.0, 1_000.0),
    ("N", "O", "SINGLE"): (55.0, 400.0),
    ("O", "O", "SINGLE"): (35.0, 250.0),
    ("C", "S", "SINGLE"): (65.0, 250.0),
    ("H", "S", "SINGLE"): (88.0, 320.0),
}

# Compatibility maps kept for older code that may import them directly.
BOND_DISS_MAP = {bond_type: params[0] for bond_type, params in BOND_ORDER_DEFAULTS.items()}
BOND_K_MAP = {bond_type: params[1] for bond_type, params in BOND_ORDER_DEFAULTS.items()}

DIHEDRAL_PARAMS = {
    Chem.rdchem.BondType.SINGLE: (0.0, 0.0, 1.5),
    Chem.rdchem.BondType.DOUBLE: (0.0, 15.0, 0.0),
    Chem.rdchem.BondType.AROMATIC: (0.0, 10.0, 0.0),
    Chem.rdchem.BondType.TRIPLE: (0.0, 0.0, 0.0),
}
ANGLE_K_HYB = {
    Chem.rdchem.HybridizationType.SP3: 100.0,
    Chem.rdchem.HybridizationType.SP2: 140.0,
    Chem.rdchem.HybridizationType.SP: 200.0,
    Chem.rdchem.HybridizationType.OTHER: 100.0,
}
EPSILON_MAP = {
    "H": 0.02,
    "C": 0.08,
    "N": 0.10,
    "O": 0.12,
    "S": 0.15,
    "F": 0.10,
    "Cl": 0.15,
    "Br": 0.20,
    "I": 0.25,
}
SIGMA_SCALE = 0.85


def _bond_label(bond_type: Chem.rdchem.BondType) -> str:
    if bond_type == Chem.rdchem.BondType.DOUBLE:
        return "DOUBLE"
    if bond_type == Chem.rdchem.BondType.TRIPLE:
        return "TRIPLE"
    if bond_type == Chem.rdchem.BondType.AROMATIC:
        return "AROMATIC"
    return "SINGLE"


def bond_morse_parameters(
    symbol_i: str, symbol_j: str, bond_type: Chem.rdchem.BondType
) -> tuple[float, float]:
    """Return Morse D_e and harmonic k for a bond.

    Returns:
        tuple: ``(D_e, k_harmonic)`` in kcal/mol and kcal/mol/Å².
    """
    a, b = sorted((symbol_i, symbol_j))
    label = _bond_label(bond_type)
    pair_params = BOND_PAIR_PARAMS.get((a, b, label))
    if pair_params is not None:
        return pair_params
    return BOND_ORDER_DEFAULTS.get(bond_type, BOND_ORDER_DEFAULTS[Chem.rdchem.BondType.SINGLE])


def get_preset(name: str) -> SimulationPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(PRESETS))
        raise ValueError(f"Preset inválido '{name}'. Use: {allowed}.") from exc
