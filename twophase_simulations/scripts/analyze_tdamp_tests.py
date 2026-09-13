#!/usr/bin/env python3
"""Summarize T and P fluctuations from thermostat damping tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations")


def parse_lammps_thermo(path: Path) -> pd.DataFrame:
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            cols = line.split()
            if len(cols) != 22:
                continue
            if cols[0] == "Step":
                header = cols
                continue
            if header is None:
                continue
            try:
                rows.append([float(x) for x in cols])
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=header)


def stats(series: pd.Series):
    return dict(
        mean=float(series.mean()),
        std=float(series.std()),
        min=float(series.min()),
        max=float(series.max()),
        ptp=float(series.max() - series.min()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=int, default=1600)
    ap.add_argument("--press", type=int, default=225)
    ap.add_argument("--tmin", type=float, default=1.0, help="discard early time (ps)")
    args = ap.parse_args()

    base = ROOT / f"p{args.press}/twophase/NPT_tdamp_tests"
    runs = sorted(base.glob(f"T{args.temp}_tdamp*"))
    if not runs:
        raise SystemExit(f"No runs under {base}")

    # expected kinetic T fluctuation for N atoms (3N dof approx)
    N = 2048
    Tset = args.temp
    sigma_theory = Tset * np.sqrt(2.0 / (3 * N))

    summary = []
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for run in runs:
        out = run / "lammps.out.init.txt"
        if not out.exists() or "Total wall time" not in out.read_text()[-500:]:
            print(f"SKIP incomplete {run.name}")
            continue
        df = parse_lammps_thermo(out)
        d = df[df["Time"] >= args.tmin]
        tag = run.name.split("tdamp")[-1].replace("p", ".")
        st = stats(d["Temp"])
        sp = stats(d["Press"] * 1e-4)  # GPa
        summary.append((tag, st, sp, len(d)))
        axes[0].plot(d["Time"], d["Temp"], lw=0.9, label=f"tdamp={tag} ps")
        axes[1].plot(d["Time"], d["Press"] * 1e-4, lw=0.9, label=f"tdamp={tag} ps")
        print(
            f"tdamp={tag:>6} ps | Temp mean={st['mean']:.1f} std={st['std']:.1f} "
            f"ptp={st['ptp']:.1f} | P mean={sp['mean']:.2f} std={sp['std']:.2f} "
            f"ptp={sp['ptp']:.2f} GPa | N={len(d)}"
        )

    print(f"\nExpected kinetic T std ~ T*sqrt(2/(3N)) = {sigma_theory:.1f} K for N={N}")
    print("(Changing tdamp mainly affects thermostat oscillations / correlation time,")
    print(" not the equilibrium variance of instantaneous kinetic temperature.)")

    axes[0].axhline(Tset, color="k", ls="--", lw=0.8)
    axes[0].axhline(Tset + sigma_theory, color="k", ls=":", lw=0.7)
    axes[0].axhline(Tset - sigma_theory, color="k", ls=":", lw=0.7)
    axes[0].set_ylabel("T (K)")
    axes[0].set_title(f"Thermostat damping tests @ {Tset} K, {args.press} GPa (after {args.tmin} ps)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(args.press, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("P (GPa)")
    axes[1].set_xlabel("Time (ps)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_png = base / f"tdamp_compare_T{args.temp}.png"
    fig.savefig(out_png, dpi=160)
    print(f"Wrote {out_png}")

    # table
    tab = base / f"tdamp_compare_T{args.temp}.txt"
    with open(tab, "w") as f:
        f.write(f"# Tset={Tset} Pset={args.press} tmin={args.tmin}ps  theory_sigma_T={sigma_theory:.2f}K\n")
        f.write("# tdamp_ps  T_mean  T_std  T_ptp  P_mean_GPa  P_std_GPa  P_ptp_GPa  N\n")
        for tag, st, sp, n in summary:
            f.write(
                f"{tag}  {st['mean']:.2f}  {st['std']:.2f}  {st['ptp']:.2f}  "
                f"{sp['mean']:.3f}  {sp['std']:.3f}  {sp['ptp']:.3f}  {n}\n"
            )
    print(f"Wrote {tab}")


if __name__ == "__main__":
    main()
