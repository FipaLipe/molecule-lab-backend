"""Bonded and non-bonded force calculations for the educational MD engine."""

from __future__ import annotations

import numpy as np

from molecule_lab.simulation.parameters import SimulationPreset
from molecule_lab.simulation.topology import Topology, lj_params


class VerletList:
    def __init__(self, r_cut: float, r_skin: float):
        self.r_cut = r_cut
        self.r_skin = r_skin
        self.pos_ref: np.ndarray | None = None
        self.pair_i = np.empty(0, dtype=np.int32)
        self.pair_j = np.empty(0, dtype=np.int32)
        self.epsilons = np.empty(0)
        self.sigmas = np.empty(0)
        self.scales = np.empty(0)
        self.charges_i = np.empty(0)
        self.charges_j = np.empty(0)

    def needs_rebuild(self, pos: np.ndarray) -> bool:
        if self.pos_ref is None:
            return True
        displacement = np.linalg.norm(pos - self.pos_ref, axis=1)
        return float(np.max(displacement)) > self.r_skin * 0.5

    def build(self, pos: np.ndarray, topology: Topology, preset: SimulationPreset) -> None:
        r_list = self.r_cut + self.r_skin
        pi_l: list[int] = []
        pj_l: list[int] = []
        eps_l: list[float] = []
        sig_l: list[float] = []
        sc_l: list[float] = []
        qi_l: list[float] = []
        qj_l: list[float] = []

        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                pair = (i, j)
                if pair in topology.bonded_pairs or pair in topology.one_three:
                    continue
                if np.linalg.norm(pos[j] - pos[i]) > r_list:
                    continue
                scale = preset.lj_scale_14 if pair in topology.one_four else 1.0
                sigma, epsilon = lj_params(
                    topology.symbols[i],
                    topology.symbols[j],
                    topology.radii[i],
                    topology.radii[j],
                )
                pi_l.append(i)
                pj_l.append(j)
                eps_l.append(epsilon)
                sig_l.append(sigma)
                sc_l.append(scale)
                qi_l.append(topology.charges[i])
                qj_l.append(topology.charges[j])

        self.pair_i = np.array(pi_l, dtype=np.int32)
        self.pair_j = np.array(pj_l, dtype=np.int32)
        self.epsilons = np.array(eps_l)
        self.sigmas = np.array(sig_l)
        self.scales = np.array(sc_l)
        self.charges_i = np.array(qi_l)
        self.charges_j = np.array(qj_l)
        self.pos_ref = pos.copy()


def nonbonded_forces(
    pos: np.ndarray, vlist: VerletList, preset: SimulationPreset
) -> tuple[np.ndarray, float, np.ndarray]:
    n = len(pos)
    forces = np.zeros((n, 3))
    atom_energy = np.zeros(n)
    if len(vlist.pair_i) == 0:
        return forces, 0.0, atom_energy

    i_idx = vlist.pair_i
    j_idx = vlist.pair_j
    delta = pos[j_idx] - pos[i_idx]
    r2 = np.einsum("ij,ij->i", delta, delta) + 1e-24
    r = np.sqrt(r2)
    mask = r <= preset.r_cut
    if not np.any(mask):
        return forces, 0.0, atom_energy

    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    delta = delta[mask]
    r = r[mask]
    r2 = r2[mask]
    scales = vlist.scales[mask]
    sigmas = vlist.sigmas[mask]
    epsilons = vlist.epsilons[mask]

    sr2 = (sigmas / r) ** 2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    v_lj = 4.0 * epsilons * (sr12 - sr6) * scales
    f_lj = 24.0 * epsilons / r * (2.0 * sr12 - sr6) * scales

    v_c = preset.k_coulomb * vlist.charges_i[mask] * vlist.charges_j[mask] / r * scales
    f_c = preset.k_coulomb * vlist.charges_i[mask] * vlist.charges_j[mask] / r2 * scales
    force_vectors = ((f_lj + f_c) / r)[:, None] * delta

    np.add.at(forces, i_idx, -force_vectors)
    np.add.at(forces, j_idx, force_vectors)

    pair_energy = v_lj + v_c
    half_energy = 0.5 * pair_energy
    np.add.at(atom_energy, i_idx, half_energy)
    np.add.at(atom_energy, j_idx, half_energy)
    return forces, float(np.sum(pair_energy)), atom_energy


def bonded_forces_and_energy(
    pos: np.ndarray, topology: Topology
) -> tuple[np.ndarray, float, np.ndarray]:
    n = len(pos)
    forces = np.zeros((n, 3))
    atom_energy = np.zeros(n)
    energy = 0.0

    if len(topology.bond_i):
        rij = pos[topology.bond_j] - pos[topology.bond_i]
        r = np.sqrt(np.einsum("ij,ij->i", rij, rij)) + 1e-12
        u = np.exp(-topology.bond_alpha * (r - topology.bond_r0))
        v = topology.bond_de * (1 - u) ** 2
        d_v_dr = 2 * topology.bond_de * topology.bond_alpha * u * (1 - u)
        fij = (d_v_dr / r)[:, None] * rij
        np.add.at(forces, topology.bond_i, fij)
        np.add.at(forces, topology.bond_j, -fij)
        half_v = 0.5 * v
        np.add.at(atom_energy, topology.bond_i, half_v)
        np.add.at(atom_energy, topology.bond_j, half_v)
        energy += float(np.sum(v))

    if len(topology.angle_i):
        rci = pos[topology.angle_i] - pos[topology.angle_c]
        rcj = pos[topology.angle_j] - pos[topology.angle_c]
        r_ci = np.sqrt(np.einsum("ij,ij->i", rci, rci)) + 1e-12
        r_cj = np.sqrt(np.einsum("ij,ij->i", rcj, rcj)) + 1e-12
        dot_ij = np.einsum("ij,ij->i", rci, rcj)
        cos_t = np.clip(dot_ij / (r_ci * r_cj), -1 + 1e-9, 1 - 1e-9)
        theta = np.arccos(cos_t)
        sin_t = np.sqrt(np.maximum(1 - cos_t**2, 1e-12))
        dth = theta - topology.angle_theta0
        v = 0.5 * topology.angle_k * dth**2
        coeff = -topology.angle_k * dth / sin_t
        fi = (coeff / r_ci)[:, None] * (
            rcj / r_cj[:, None] - cos_t[:, None] * rci / r_ci[:, None]
        )
        fj = (coeff / r_cj)[:, None] * (
            rci / r_ci[:, None] - cos_t[:, None] * rcj / r_cj[:, None]
        )
        np.add.at(forces, topology.angle_i, fi)
        np.add.at(forces, topology.angle_j, fj)
        np.add.at(forces, topology.angle_c, -(fi + fj))
        third_v = v / 3
        np.add.at(atom_energy, topology.angle_i, third_v)
        np.add.at(atom_energy, topology.angle_c, third_v)
        np.add.at(atom_energy, topology.angle_j, third_v)
        energy += float(np.sum(v))

    if len(topology.dihedral_i):
        i_idx = topology.dihedral_i
        j_idx = topology.dihedral_j
        k_idx = topology.dihedral_k
        l_idx = topology.dihedral_l
        b1 = pos[j_idx] - pos[i_idx]
        b2 = pos[k_idx] - pos[j_idx]
        b3 = pos[l_idx] - pos[k_idx]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1n = np.sqrt(np.einsum("ij,ij->i", n1, n1)) + 1e-12
        n2n = np.sqrt(np.einsum("ij,ij->i", n2, n2)) + 1e-12
        b2n = np.sqrt(np.einsum("ij,ij->i", b2, b2)) + 1e-12
        n1h = n1 / n1n[:, None]
        b2h = b2 / b2n[:, None]
        n2h = n2 / n2n[:, None]
        m1 = np.cross(n1h, b2h)
        cos_phi = np.clip(np.einsum("ij,ij->i", n1h, n2h), -1 + 1e-9, 1 - 1e-9)
        sin_phi = np.einsum("ij,ij->i", m1, n2h)
        phi = np.arctan2(sin_phi, cos_phi)
        v1 = topology.dihedral_v1
        v2 = topology.dihedral_v2
        v3 = topology.dihedral_v3
        v = (
            v1 / 2 * (1 + np.cos(phi))
            + v2 / 2 * (1 - np.cos(2 * phi))
            + v3 / 2 * (1 + np.cos(3 * phi))
        )
        d_v_dp = (
            -v1 / 2 * np.sin(phi)
            + v2 * np.sin(2 * phi)
            - 3 * v3 / 2 * np.sin(3 * phi)
        )
        active = np.abs(d_v_dp) >= 1e-12
        if np.any(active):
            fi = ((d_v_dp * b2n / n1n**2)[:, None]) * n1
            fl = -((d_v_dp * b2n / n2n**2)[:, None]) * n2
            b2l2 = b2n**2
            b1b2 = np.einsum("ij,ij->i", b1, b2)
            b3b2 = np.einsum("ij,ij->i", b3, b2)
            fj = ((b1b2 / b2l2 - 1)[:, None]) * fi - (
                (b3b2 / b2l2)[:, None]
            ) * fl
            fk = -(fi + fj + fl)
            np.add.at(forces, i_idx[active], fi[active])
            np.add.at(forces, j_idx[active], fj[active])
            np.add.at(forces, k_idx[active], fk[active])
            np.add.at(forces, l_idx[active], fl[active])
        quarter_v = v / 4
        np.add.at(atom_energy, i_idx, quarter_v)
        np.add.at(atom_energy, j_idx, quarter_v)
        np.add.at(atom_energy, k_idx, quarter_v)
        np.add.at(atom_energy, l_idx, quarter_v)
        energy += float(np.sum(v))

    return forces, energy, atom_energy


def forces_and_energy(
    pos: np.ndarray,
    topology: Topology,
    vlist: VerletList,
    preset: SimulationPreset,
) -> tuple[np.ndarray, float, np.ndarray]:
    bonded_forces, bonded_energy, bonded_atom_energy = bonded_forces_and_energy(
        pos, topology
    )
    nonbonded_force, nonbonded_energy, nonbonded_atom_energy = nonbonded_forces(
        pos, vlist, preset
    )
    return (
        bonded_forces + nonbonded_force,
        bonded_energy + nonbonded_energy,
        bonded_atom_energy + nonbonded_atom_energy,
    )
