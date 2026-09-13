#!/usr/bin/env python3
"""
Ricky-style z-slab solid/liquid map, retuned for *atomic* hydrogen.

Idea
----
Keep the same geometry as the molecular Ricky script:
  - split the box into `num1` coarse slabs along z
  - within each slab, build a fine histogram with `num2` bins
  - score each slab every frame -> ternary map (solid / interface / liquid)

For molecular H2 the score was raw std of fine-bin counts (high when dimers
form tight layers). For atomic H that absolute std depends on density and is
weakly separating. Instead use a *normalized* layering score:

  local CV  =  std(h) / mean(h)
  peak-to-valley (P2V) = (max(h) - min(h)) / (max(h) + min(h))

Solid layers: atoms pile into a few fine bins -> high CV / high P2V.
Atomic liquid: flatter occupation across fine bins -> low CV / low P2V.

Because the score is computed *per slab per frame*, the solid/liquid boundary
can migrate in z as the interface melts or freezes — no fixed z window.

Calibration on pure 1024-atom cells @ 1600 K / 225 GPa (num1=10, num2=20):
  solid  CV ~ 0.61–0.88
  liquid CV ~ 0.30–0.50
Default thresholds place an interface band between them.

Writes under analysis/:
  P{P}T{T}_state_cv.txt, _metric_cv.txt, _density_cv.txt, P{P}T{T}fig_cv.png
  plus a side-by-side old-std vs new-CV comparison png
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import io
from matplotlib.colors import ListedColormap

ROOT = "/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations"


def gen_data_path(P, T, phase="twophase", case="NPT_mliap"):
    return f"{ROOT}/p{P}/{phase}/{case}/{T}/analysis/"


def fine_hist(z, Lz, num1, num2):
    fine = num1 * num2
    hist, _ = np.histogram(np.mod(z, Lz), bins=np.arange(0.0, Lz + Lz / fine, Lz / fine))
    return hist.astype(float)


def slab_scores(hist, num1, num2, metric="cv"):
    """Return (num1,) score array for one frame."""
    scores = np.zeros(num1)
    for j in range(num1):
        h = hist[j * num2 : (j + 1) * num2]
        mean = h.mean()
        std = h.std()
        if metric == "cv":
            scores[j] = std / (mean + 1e-12)
        elif metric == "p2v":
            scores[j] = (h.max() - h.min()) / (h.max() + h.min() + 1e-12)
        elif metric == "std":
            scores[j] = std
        else:
            raise ValueError(metric)
    return scores


def classify(scores, t_solid, t_liquid, metric="cv"):
    """
    state: 0=solid, 1=interface, 2=liquid
    For cv/p2v/std: higher => more solid-like.
    """
    state = np.ones(len(scores), dtype=int)  # interface default
    state[scores >= t_solid] = 0
    state[scores <= t_liquid] = 2
    return state


def proportions_and_rs(hist, state, volume, num1, num2):
    solid_part = np.where(state == 0)[0]
    liquid_part = np.where(state == 2)[0]
    interface_part = np.where(state == 1)[0]
    solid_density = sum(hist[j * num2 : (j + 1) * num2].sum() for j in solid_part)
    liquid_density = sum(hist[j * num2 : (j + 1) * num2].sum() for j in liquid_part)
    row = np.full(5, np.nan)
    if len(solid_part):
        rho = solid_density / (volume / num1 * len(solid_part))
        row[1] = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0) / 0.529
    if len(liquid_part):
        rho = liquid_density / (volume / num1 * len(liquid_part))
        row[2] = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0) / 0.529
    row[3] = (len(solid_part) + 0.5 * len(interface_part)) / num1
    row[4] = (len(liquid_part) + 0.5 * len(interface_part)) / num1
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--press", type=int, required=True)
    parser.add_argument("-t", "--temp", type=int, required=True)
    parser.add_argument("-c", "--case", default="NPT_mliap")
    parser.add_argument("-f", "--phase", default="twophase")
    parser.add_argument("--num1", type=int, default=15, help="coarse slabs along z")
    parser.add_argument("--num2", type=int, default=20, help="fine bins per slab")
    parser.add_argument(
        "--metric",
        choices=("cv", "p2v", "std"),
        default="cv",
        help="layering score (default: local CV)",
    )
    # CV defaults from pure-phase calibration (gap between liquid max~0.50 and solid min~0.61)
    parser.add_argument("--t-solid", type=float, default=None, help="score >= this => solid")
    parser.add_argument("--t-liquid", type=float, default=None, help="score <= this => liquid")
    parser.add_argument("--frame2time", type=float, default=0.05)
    parser.add_argument("--step2time", type=float, default=0.0005)
    parser.add_argument(
        "--also-old-std",
        action="store_true",
        default=True,
        help="also compute old molecular std map for comparison (default on)",
    )
    args = parser.parse_args()

    # Default thresholds by metric
    defaults = {
        "cv": (0.58, 0.50),   # solid >= 0.58, liquid <= 0.50
        "p2v": (0.90, 0.80),
        "std": (3.2, 2.5),    # retuned absolute std for atomic (old was 5/4)
    }
    t_solid, t_liquid = defaults[args.metric]
    if args.t_solid is not None:
        t_solid = args.t_solid
    if args.t_liquid is not None:
        t_liquid = args.t_liquid

    path = gen_data_path(args.press, args.temp, args.phase, args.case)
    dump = os.path.join(path, "dump.merged.atom")
    csv = os.path.join(path, f"P{args.press}T{args.temp}M29_merged_lammps_out.csv")
    if not os.path.isfile(dump):
        raise FileNotFoundError(dump)
    if not os.path.isfile(csv):
        raise FileNotFoundError(csv)

    frames = io.read(dump, format="lammps-dump-text", index=":")
    if not isinstance(frames, list):
        frames = [frames]
    thermo = pd.read_csv(csv)
    nframes = len(frames)
    num1, num2 = args.num1, args.num2

    metric_map = np.zeros((nframes, num1))
    state = np.zeros((nframes, num1), dtype=int)
    density = np.zeros((nframes, 5))

    # old molecular-style std map for comparison
    state_old = np.zeros((nframes, num1), dtype=int)
    std_map = np.zeros((nframes, num1))

    for i, atoms in enumerate(frames):
        atoms.set_pbc(True)
        atoms.wrap()
        z = atoms.get_positions()[:, 2]
        Lz = float(atoms.get_cell()[2, 2])
        hist = fine_hist(z, Lz, num1, num2)
        scores = slab_scores(hist, num1, num2, args.metric)
        metric_map[i] = scores
        state[i] = classify(scores, t_solid, t_liquid, args.metric)
        density[i] = proportions_and_rs(hist, state[i], atoms.get_volume(), num1, num2)
        density[i, 0] = i

        stds = slab_scores(hist, num1, num2, "std")
        std_map[i] = stds
        # original Ricky molecular thresholds
        state_old[i] = classify(stds, t_solid=5.0, t_liquid=4.0)

    tag = args.metric
    np.savetxt(os.path.join(path, f"P{args.press}T{args.temp}_state_{tag}.txt"), state, fmt="%d")
    np.savetxt(os.path.join(path, f"P{args.press}T{args.temp}_metric_{tag}.txt"), metric_map)
    np.savetxt(os.path.join(path, f"P{args.press}T{args.temp}_density_{tag}.txt"), density)

    # --- main figure (Ricky layout) ---
    frame2time = args.frame2time
    step2time = args.step2time
    cmap_state = ListedColormap(["#d62728", "#bcbd22", "#1f77b4"])  # solid, interface, liquid

    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(7.5, 9.5))
    im0 = axes[0].imshow(
        state.T,
        aspect="auto",
        origin="lower",
        cmap=cmap_state,
        vmin=0,
        vmax=2,
        extent=[0, nframes * frame2time, 0, num1],
        interpolation="nearest",
    )
    axes[0].set_ylabel("slab")
    axes[0].set_title(f"State map ({tag}: solid≥{t_solid}, liquid≤{t_liquid})")
    axes[0].set_xticks([])
    cbar = fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["solid", "iface", "liquid"])

    im1 = axes[1].imshow(
        metric_map.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, nframes * frame2time, 0, num1],
        interpolation="nearest",
    )
    axes[1].set_ylabel("slab")
    axes[1].set_title(f"Raw {tag} score (high = layered / solid-like)")
    axes[1].set_xticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

    axes[2].plot(density[:, 0] * frame2time, density[:, 3], label="solid")
    axes[2].plot(density[:, 0] * frame2time, density[:, 4], label="liquid")
    axes[2].set_xlim(0, nframes * frame2time)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Proportion")
    axes[2].set_xticks([])
    axes[2].legend(fontsize=8)

    axes[3].plot(density[:, 0] * frame2time, density[:, 1], label="solid")
    axes[3].plot(density[:, 0] * frame2time, density[:, 2], label="liquid")
    axes[3].set_xlim(0, nframes * frame2time)
    axes[3].set_ylabel(r"$r_s$")
    axes[3].set_xticks([])
    axes[3].legend(fontsize=8)

    bar2gpa = 1e-4
    axes[4].plot(thermo["Step"] * step2time, thermo["Pxx"] * bar2gpa, label="pxx")
    axes[4].plot(thermo["Step"] * step2time, thermo["Pyy"] * bar2gpa, label="pyy")
    axes[4].plot(thermo["Step"] * step2time, thermo["Pzz"] * bar2gpa, label="pzz")
    axes[4].set_xlim(0, nframes * frame2time)
    axes[4].set_ylabel(r"$P_d$ (GPa)")
    axes[4].set_xticks([])
    axes[4].legend(fontsize=7)

    axes[5].plot(thermo["Step"] * step2time, thermo["Pxy"] * bar2gpa, label="pxy")
    axes[5].plot(thermo["Step"] * step2time, thermo["Pxz"] * bar2gpa, label="pxz")
    axes[5].plot(thermo["Step"] * step2time, thermo["Pyz"] * bar2gpa, label="pyz")
    axes[5].set_xlim(0, nframes * frame2time)
    axes[5].set_ylabel(r"$P_{od}$ (GPa)")
    axes[5].set_xticks([])
    axes[5].legend(fontsize=7)

    axes[6].plot(thermo["Step"] * step2time, thermo["Lx"], label="lx")
    axes[6].plot(thermo["Step"] * step2time, thermo["Ly"], label="ly")
    axes[6].plot(thermo["Step"] * step2time, thermo["Lz"], label="lz")
    axes[6].set_xlim(0, nframes * frame2time)
    axes[6].set_ylabel(r"L (Å)")
    axes[6].set_xticks([])
    axes[6].legend(fontsize=7)

    axes[7].plot(thermo["Step"] * step2time, thermo["Temp"])
    axes[7].set_xlim(0, nframes * frame2time)
    axes[7].set_xlabel("Time (ps)")
    axes[7].set_ylabel("T (K)")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    out_fig = os.path.join(path, f"P{args.press}T{args.temp}fig_{tag}.png")
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_fig}")

    # --- comparison: old molecular std vs new metric ---
    if args.also_old_std:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes[0, 0].imshow(
            state_old.T,
            aspect="auto",
            origin="lower",
            cmap=cmap_state,
            vmin=0,
            vmax=2,
            extent=[0, nframes * frame2time, 0, num1],
            interpolation="nearest",
        )
        axes[0, 0].set_title("OLD molecular std map (ts=5, tl=4)")
        axes[0, 0].set_ylabel("slab")
        axes[0, 1].imshow(
            state.T,
            aspect="auto",
            origin="lower",
            cmap=cmap_state,
            vmin=0,
            vmax=2,
            extent=[0, nframes * frame2time, 0, num1],
            interpolation="nearest",
        )
        axes[0, 1].set_title(f"NEW {tag} map (solid≥{t_solid}, liquid≤{t_liquid})")
        axes[1, 0].imshow(
            std_map.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, nframes * frame2time, 0, num1],
            interpolation="nearest",
        )
        axes[1, 0].set_title("Raw std score")
        axes[1, 0].set_xlabel("Time (ps)")
        axes[1, 0].set_ylabel("slab")
        im = axes[1, 1].imshow(
            metric_map.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, nframes * frame2time, 0, num1],
            interpolation="nearest",
        )
        axes[1, 1].set_title(f"Raw {tag} score")
        axes[1, 1].set_xlabel("Time (ps)")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
        fig.suptitle(f"P={args.press} GPa  T={args.temp} K — slab map retune", fontsize=11)
        fig.tight_layout()
        out_cmp = os.path.join(path, f"P{args.press}T{args.temp}_compare_oldstd_vs_{tag}.png")
        fig.savefig(out_cmp, dpi=180)
        plt.close(fig)
        print(f"Wrote {out_cmp}")

    # summary
    print(
        f"T={args.temp}: mean solid frac={np.nanmean(density[:,3]):.3f}, "
        f"liquid frac={np.nanmean(density[:,4]):.3f}, "
        f"metric[{tag}] mean={metric_map.mean():.3f} "
        f"range=[{metric_map.min():.3f},{metric_map.max():.3f}]"
    )


if __name__ == "__main__":
    main()
