#!/usr/bin/env python3
"""Compute shell-averaged S(k) for equilibrated solid/liquid 1536-atom supercells."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen")
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack")
from qharv_db.grsk import calc_sofk_npt


def load_traj(path: Path, index: str = ":"):
    traj = read(str(path), format="lammps-dump-text", index=index)
    if not isinstance(traj, list):
        traj = [traj]
    axesl, posl = [], []
    for atoms in traj:
        atoms.set_pbc(True)
        axesl.append(np.array(atoms.get_cell()))
        posl.append(atoms.get_positions())
    return axesl, posl, traj[0]


def compute_sofk(axesl, posl, nsh: int = 16, kmax: float = 8.0):
    L_box = max(float(np.diag(ax).max()) for ax in axesl)
    dk = 2.0 * np.pi / L_box
    bin_edges = np.arange(1e-3, kmax + 1e-8, dk)
    uskm, uske = calc_sofk_npt(bin_edges, axesl, posl, nsh=nsh)
    delta = np.diff(bin_edges)
    k = bin_edges[: len(uskm)] + delta[: len(uskm)] / 2.0
    return k, uskm, uske, L_box


def summarize(phase: str, k, sk):
    # Ignore very small k (finite-size / k=0 noise)
    mask = k > 0.5
    kk, ss = k[mask], sk[mask]
    imax = int(np.argmax(ss))
    # Secondary: tallest peak with k > first_peak + 0.5 (crystalline higher-order)
    k1, s1 = float(kk[imax]), float(ss[imax])
    mask2 = kk > k1 + 0.5
    if np.any(mask2):
        i2 = int(np.argmax(ss[mask2]))
        k2, s2 = float(kk[mask2][i2]), float(ss[mask2][i2])
    else:
        k2, s2 = np.nan, 0.0
    print(f"[{phase}] nframes-averaged S(k):")
    print(f"  max peak: S={s1:.2f} at k={k1:.3f} A^-1")
    print(f"  next peak (k>k1+0.5): S={s2:.2f} at k={k2:.3f} A^-1")
    return s1, s2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations/p225",
    )
    parser.add_argument("--nsh", type=int, default=16)
    parser.add_argument("--kmax", type=float, default=8.0)
    parser.add_argument(
        "--index",
        default="-10:",
        help="ASE frame slice (default: last 10 frames of 1 ps dump)",
    )
    args = parser.parse_args()
    root = Path(args.root)
    outdir = root / "sofk_equilibrated_1600K"
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "solid": root / "solid_1536/equilibrate_1600K/dump.equilibrate.atom",
        "liquid": root / "liquid_1536/equilibrate_1600K/dump.equilibrate.atom",
    }

    results = {}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for phase, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        axesl, posl, atoms0 = load_traj(path, index=args.index)
        print(f"[{phase}] {len(axesl)} frames, N={len(atoms0)}, cell={atoms0.cell.lengths()}")
        k, sk, ske, L_box = compute_sofk(axesl, posl, nsh=args.nsh, kmax=args.kmax)
        np.savetxt(
            outdir / f"sofk_{phase}_1536_1600K.dat",
            np.column_stack([k, sk, ske]),
            header="k(A^-1) S(k) sigma",
        )
        s1, s2 = summarize(phase, k, sk)
        results[phase] = (s1, s2, L_box)
        ax.errorbar(k, sk, yerr=ske, fmt="-o", ms=3, lw=1.2, label=f"{phase} (N=1536)")

    ax.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$S(k)$")
    ax.set_title("Equilibrated 1536-atom supercells @ 1600 K, 225 GPa density")
    ax.set_xlim(0, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "sofk_solid_vs_liquid_1536_1600K.png", dpi=160)
    print(f"Wrote {outdir}/sofk_solid_vs_liquid_1536_1600K.png")

    # Simple verdict: crystalline solid has tall Bragg peak(s); liquid S(k) ~ O(1-3)
    s_sol, s2_sol, _ = results["solid"]
    s_liq, s2_liq, _ = results["liquid"]
    solid_ok = s_sol > 10
    liquid_ok = s_liq < 5
    print("--- verdict ---")
    print(
        f"solid:   {'SOLID-like (sharp Bragg peak)' if solid_ok else 'UNCERTAIN'} "
        f"(Smax={s_sol:.1f}, S2={s2_sol:.1f})"
    )
    print(
        f"liquid:  {'LIQUID-like (no sharp Bragg peaks)' if liquid_ok else 'UNCERTAIN'} "
        f"(Smax={s_liq:.1f}, S2={s2_liq:.1f})"
    )
    if not (solid_ok and liquid_ok):
        sys.exit(2)


if __name__ == "__main__":
    main()
