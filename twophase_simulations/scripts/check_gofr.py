#!/usr/bin/env python3
"""Compute g(r) from a LAMMPS dump/data and check for molecular (dimer) peak.

Atomic liquid at ~225 GPa should NOT have a sharp first peak near ~0.74 A.
A significant g(r) below --r-mol-max indicates residual molecular character.
"""
from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/mysharelib")
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen")
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack")


def get_gofr_norm(axes, bin_edges, n1):
    ndim = axes.shape[0]
    vnorm = np.diff(2 * (ndim - 1) / ndim * np.pi * bin_edges**ndim)
    npair = n1 * (n1 - 1) / 2
    rho = npair / abs(np.linalg.det(axes))
    return 1.0 / (rho * vnorm)


def ase_gofr(atoms, bin_edges):
    from ase.geometry import get_distances

    pos = atoms.get_positions()
    _, rij = get_distances(pos, p2=pos, cell=atoms.get_cell(), pbc=True)
    idx = np.triu_indices_from(rij, 1)
    dists = rij[idx]
    hist, _ = np.histogram(dists, bin_edges)
    gr_norm = get_gofr_norm(np.array(atoms.get_cell()), bin_edges, len(pos))
    return hist * gr_norm


def read_frames(path, index):
    path_l = path.lower()
    if "dump" in path_l or path_l.endswith(".atom"):
        # Support ASE slice strings: ":", "-20:", "0:10:2", or a single int
        if index in (":", "all") or ":" in str(index):
            frames = read(path, index=index if index != "all" else ":", format="lammps-dump-text")
            return list(frames) if isinstance(frames, list) else [frames]
        return [read(path, index=int(index), format="lammps-dump-text")]
    return [read(path, format="lammps-data", atom_style="atomic", sort_by_id=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-prefix", default="gofr")
    parser.add_argument("--index", default="-1", help="Frame index, or ':' to average all")
    parser.add_argument("--dr", type=float, default=0.02)
    parser.add_argument("--rmax", type=float, default=None)
    parser.add_argument(
        "--r-mol-max",
        type=float,
        default=0.85,
        help="Upper edge of molecular-dimer window (A); classic H2 peak ~0.74 A",
    )
    parser.add_argument(
        "--mol-peak-thresh",
        type=float,
        default=1.5,
        help="Max g(r) allowed in dimer window for 'atomic' pass "
        "(high-P atomic liquids can have g~0.5-1 rising into the 1st shell)",
    )
    args = parser.parse_args()

    frames = read_frames(args.input, args.index)
    if not isinstance(frames, list):
        frames = list(frames)
    for a in frames:
        a.set_pbc(True)

    # Use last frame cell for binning if single; else min half-box
    cells = [np.array(a.get_cell()) for a in frames]
    half = min(0.5 * np.linalg.norm(c, axis=1).min() for c in cells)
    rmax = args.rmax if args.rmax is not None else half
    bin_edges = np.arange(0.0, rmax + args.dr, args.dr)
    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    grs = []
    for a in frames:
        grs.append(ase_gofr(a, bin_edges))
    gr = np.mean(grs, axis=0)

    np.savez(f"{args.output_prefix}.npz", r=r, gr=gr, bin_edges=bin_edges)
    np.savetxt(
        f"{args.output_prefix}.dat",
        np.column_stack([r, gr]),
        header="r(A) g(r)",
    )

    mol_mask = r < args.r_mol_max
    mol_max = float(gr[mol_mask].max()) if np.any(mol_mask) else 0.0
    mol_r = float(r[mol_mask][np.argmax(gr[mol_mask])]) if np.any(mol_mask) else 0.0
    # first peak overall
    # ignore very small r noise
    valid = r > 0.3
    peak_idx = np.argmax(gr[valid])
    peak_r = float(r[valid][peak_idx])
    peak_g = float(gr[valid][peak_idx])

    is_atomic = mol_max < args.mol_peak_thresh
    verdict = "ATOMIC" if is_atomic else "MOLECULAR/RESIDUAL-DIMERS"
    print(f"Frames averaged: {len(frames)}")
    print(f"First peak: r={peak_r:.3f} A, g={peak_g:.3f}")
    print(f"Molecular window r<{args.r_mol_max}: max g={mol_max:.3f} at r={mol_r:.3f}")
    print(f"Verdict: {verdict} (thresh g<{args.mol_peak_thresh})")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(r, gr, "k-", lw=1.5)
    ax.axvline(args.r_mol_max, color="C1", ls="--", label=f"mol window < {args.r_mol_max} A")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_title(f"g(r) — {verdict}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(rmax, 3.0))
    fig.tight_layout()
    fig.savefig(f"{args.output_prefix}.png", dpi=150)
    print(f"Wrote {args.output_prefix}.dat/.npz/.png")

    if not is_atomic:
        sys.exit(2)


if __name__ == "__main__":
    main()
