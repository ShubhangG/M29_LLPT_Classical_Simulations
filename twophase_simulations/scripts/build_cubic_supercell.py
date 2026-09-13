#!/usr/bin/env python3
"""Build an nx x ny x nz 'lego' supercell from a single LAMMPS data / ASE-readable config.

Default 2x2x2: four cells in a square on the bottom layer, four on top.
Optionally leave a z-gap between the two layers (plus a matching PBC gap)
so a subsequent deform relax can close the seams.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import read

sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/mysharelib")
from mytool import myio


def read_atoms(path: str, index: str | int = -1) -> Atoms:
    path_l = path.lower()
    if path_l.endswith(".xyz"):
        atoms = read(path, index=index)
    elif "dump" in path_l or path_l.endswith(".atom"):
        atoms = read(path, index=index, format="lammps-dump-text")
    else:
        # LAMMPS data written by myio
        atoms = read(path, format="lammps-data", atom_style="atomic", sort_by_id=True)
    atoms.set_pbc(True)
    return atoms


def build_lego_supercell(
    atoms: Atoms,
    nx: int = 2,
    ny: int = 2,
    nz: int = 2,
    gap: float = 1.0,
) -> Atoms:
    """Tile atoms into an nx*ny*nz supercell.

    For nz >= 2 and gap > 0, the z-layers are separated by `gap` (Angstrom),
    and the box includes an extra `gap` at the PBC boundary so a deform of
    `-2*gap` closes both seams (same convention as the old two-phase slab).
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz must be >= 1")

    layer = make_supercell(atoms, np.diag([nx, ny, 1]))
    if nz == 1 or gap <= 0.0:
        return make_supercell(atoms, np.diag([nx, ny, nz]))

    lz = float(layer.cell[2, 2])
    positions = []
    for iz in range(nz):
        pos = layer.get_positions().copy()
        pos[:, 2] += iz * (lz + gap)
        positions.append(pos)
    positions = np.vstack(positions)
    cell = layer.cell.copy()
    cell[2, 2] = nz * lz + nz * gap  # (nz layers) + (nz gaps: nz-1 interior + 1 PBC)
    # For nz=2: 2*lz + 2*gap  -> deform -2*gap closes both gaps
    return Atoms(
        symbols=["H"] * len(positions),
        positions=positions,
        cell=cell,
        pbc=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Input LAMMPS data / dump / xyz")
    parser.add_argument("-o", "--output", required=True, help="Output LAMMPS data file")
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument(
        "--gap",
        type=float,
        default=1.0,
        help="Z gap (A) between layers; use 0 for seamless ASE make_supercell",
    )
    parser.add_argument("--index", default=-1, help="Frame index for dump/xyz inputs")
    args = parser.parse_args()

    index = int(args.index)
    atoms = read_atoms(args.input, index=index)
    print(f"Input: {len(atoms)} atoms, cell={atoms.cell.lengths()}")

    sc = build_lego_supercell(atoms, nx=args.nx, ny=args.ny, nz=args.nz, gap=args.gap)
    myio.atoms2data(sc, args.output, ["H"])
    print(
        f"Wrote {args.output}: {len(sc)} atoms, cell={sc.cell.lengths()}, "
        f"gap={args.gap} A, tile={args.nx}x{args.ny}x{args.nz}"
    )
    if args.gap > 0 and args.nz >= 2:
        print(f"Suggested deform close: z delta 0 {-args.nz * args.gap:.4f}")


if __name__ == "__main__":
    main()
