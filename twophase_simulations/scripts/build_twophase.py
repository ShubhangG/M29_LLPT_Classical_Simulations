#!/usr/bin/env python3
"""Stack a solid supercell (bottom) and liquid supercell (top) with a z slab gap.

Mirrors the old gen_two_phases_model.py convention:
  cell_z = Lz_solid + Lz_liquid + 2*slab
  liquid shifted by Lz_solid + slab
so a LAMMPS `fix deform ... z delta 0 -2*slab` closes both the interface
and the periodic image gap.

IMPORTANT: LAMMPS write_data files often store unwrapped coords + image flags.
We always wrap into the primary cell after reading, otherwise stacking creates
catastrophic overlaps (and Kokkos CUDA illegal-address crashes).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/mysharelib")
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/dense_hydrogen")
from dhtool import bond
from mytool import myio


def find_dimers(rij, rmax=np.inf, rmin=0, sort_id=False):
    natom, natom1 = rij.shape
    assert natom1 == natom
    found = np.zeros(natom, dtype=bool)
    pairs = []
    idx = np.triu_indices(natom, 1)
    dists = rij[idx]
    ij = np.array(idx).T
    for idist in np.argsort(dists):
        i, j = ij[idist]
        if found[i] or found[j]:
            continue
        rb = dists[idist]
        if (rb < rmin) or (rb > rmax):
            continue
        pair = (i, j) if i < j else (j, i)
        pairs.append(pair)
        found[i] = True
        found[j] = True
        if np.all(found):
            break
    pa = np.array(pairs)
    if sort_id and len(pa):
        pa = pa[np.argsort(pa[:, 0])]
    return pa


def read_atoms(path: str, index: int = -1) -> Atoms:
    path_l = path.lower()
    if "dump" in path_l or path_l.endswith(".atom"):
        atoms = read(path, index=index, format="lammps-dump-text")
    else:
        # LAMMPS write_data may be "atomic" or "atomic/kk" with image flags
        try:
            atoms = read(path, format="lammps-data", atom_style="atomic", sort_by_id=True)
        except Exception:
            atoms = read(path, format="lammps-data", sort_by_id=True)
    atoms.set_pbc(True)
    # Critical: fold image-flag-unwrapped coords back into the box
    atoms.wrap()
    return atoms


def unwrap_dimers(atoms: Atoms, rmax: float = 1.2) -> Atoms:
    """Move PBC-split molecular dimers wholly above the box (flag='u')."""
    pairs = find_dimers(atoms.get_all_distances(mic=True), rmax=rmax, sort_id=True)
    if len(pairs) == 0:
        return atoms
    return bond.move_bonded_atoms(atoms, np.array(pairs), "u")


def min_pair_distance(atoms: Atoms, cutoff: float = 2.0) -> float:
    d = neighbor_list("d", atoms, cutoff)
    return float(d.min()) if len(d) else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solid", required=True, help="Solid LAMMPS data or dump")
    parser.add_argument("--liquid", required=True, help="Liquid LAMMPS data or dump")
    parser.add_argument("-o", "--output", required=True, help="Output two-phase data")
    parser.add_argument("--slab", type=float, default=4.0, help="Interface / PBC gap (A)")
    parser.add_argument("--solid-index", type=int, default=-1)
    parser.add_argument("--liquid-index", type=int, default=-1)
    parser.add_argument(
        "--unwrap-solid",
        action="store_true",
        default=True,
        help="Unwrap molecular dimers in solid before stacking (default on)",
    )
    parser.add_argument("--no-unwrap-solid", action="store_false", dest="unwrap_solid")
    parser.add_argument(
        "--unwrap-liquid",
        action="store_true",
        default=False,
        help="Also unwrap dimers in liquid (usually off for atomic liquid)",
    )
    parser.add_argument(
        "--min-dist-abort",
        type=float,
        default=0.35,
        help="Abort if any pair closer than this (A) after stacking",
    )
    args = parser.parse_args()

    solid = read_atoms(args.solid, args.solid_index)
    liquid = read_atoms(args.liquid, args.liquid_index)
    print(f"Solid after wrap: N={len(solid)}, cell={solid.cell.lengths()}, min_d={min_pair_distance(solid):.3f}")
    print(f"Liquid after wrap: N={len(liquid)}, cell={liquid.cell.lengths()}, min_d={min_pair_distance(liquid):.3f}")

    # Lateral boxes should match; use solid lateral cell, warn if mismatch.
    sx, sy, sz = solid.cell.lengths()
    lx, ly, lz = liquid.cell.lengths()
    if abs(sx - lx) > 1e-2 or abs(sy - ly) > 1e-2:
        print(
            f"WARNING: lateral mismatch solid ({sx:.4f},{sy:.4f}) vs "
            f"liquid ({lx:.4f},{ly:.4f}); rescaling liquid xy to solid."
        )
        scale = np.array([sx / lx, sy / ly, 1.0])
        pos = liquid.get_positions()
        pos[:, 0] *= scale[0]
        pos[:, 1] *= scale[1]
        cell = liquid.cell.copy()
        cell[0] *= scale[0]
        cell[1] *= scale[1]
        liquid = Atoms(symbols=liquid.get_chemical_symbols(), positions=pos, cell=cell, pbc=True)
        liquid.wrap()
        lx, ly, lz = liquid.cell.lengths()

    if args.unwrap_solid:
        solid = unwrap_dimers(solid)
        # Keep xy folded; allow z slightly above Lz into the interface gap
        pos = solid.get_positions()
        pos[:, 0] = np.mod(pos[:, 0], sx)
        pos[:, 1] = np.mod(pos[:, 1], sy)
        solid = Atoms(symbols=solid.get_chemical_symbols(), positions=pos, cell=solid.cell, pbc=True)
        myio.atoms2data(solid, args.output.replace(".txt", "_solid_unwrapped.txt"), ["H"])
    if args.unwrap_liquid:
        liquid = unwrap_dimers(liquid)
        liquid.wrap()

    pos_s = solid.get_positions().copy()
    pos_l = liquid.get_positions().copy()
    pos_l[:, 2] += sz + args.slab
    positions = np.vstack([pos_s, pos_l])
    cell = np.diag([sx, sy, sz + lz + 2.0 * args.slab])
    twophase = Atoms(symbols=["H"] * len(positions), positions=positions, cell=cell, pbc=True)

    dmin = min_pair_distance(twophase, cutoff=2.0)
    print(f"Stacked min pair distance = {dmin:.4f} A")
    if not np.isnan(dmin) and dmin < args.min_dist_abort:
        raise RuntimeError(
            f"Stacked config has pairs closer than {args.min_dist_abort} A "
            f"(min={dmin:.4f}). Refusing to write — would crash LAMMPS/Kokkos."
        )

    myio.atoms2data(twophase, args.output, ["H"])
    print(
        f"Wrote {args.output}: {len(twophase)} atoms "
        f"(solid {len(solid)} + liquid {len(liquid)}), "
        f"cell={twophase.cell.lengths()}, slab={args.slab} A"
    )
    print(f"Suggested deform close: z delta 0 {-2.0 * args.slab:.4f}")


if __name__ == "__main__":
    main()
