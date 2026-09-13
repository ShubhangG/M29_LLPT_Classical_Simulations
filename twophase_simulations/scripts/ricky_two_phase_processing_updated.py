#!/usr/bin/env python3
"""
Two-phase solid/liquid fraction diagnostics (adapted from M18 ricky script).
Expects under analysis/:
  dump.merged.atom
  P{P}T{T}M29_merged_lammps_out.csv

Writes:
  P{P}T{T}_state.txt, P{P}T{T}_density.txt, P{P}T{T}fig.png
"""
from ase import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

ROOT = "/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations"


def gen_data_path(P, T, phase="twophase", case="NPT_mliap"):
    return f"{ROOT}/p{P}/{phase}/{case}/{T}/analysis/"


def main():
    parser = argparse.ArgumentParser(description="Two-phase NPT post-processing")
    parser.add_argument("-p", "--press", dest="pressure", type=int, required=True)
    parser.add_argument("-t", "--temp", dest="temp", type=int, required=True)
    parser.add_argument("-c", "--case", default="NPT_mliap")
    parser.add_argument("-f", "--phase", default="twophase")
    # dump every 100 steps, dt=0.0005 ps -> 0.05 ps/frame
    parser.add_argument("--frame2time", type=float, default=0.05)
    parser.add_argument("--step2time", type=float, default=0.0005)
    args = parser.parse_args()

    press = args.pressure
    temp = args.temp
    num1 = 15
    num2 = 20
    ts = 5.0
    tl = 4.0

    path = gen_data_path(press, temp, args.phase, args.case)
    dump = path + "dump.merged.atom"
    csv = path + f"P{press}T{temp}M29_merged_lammps_out.csv"
    if not os.path.isfile(dump):
        raise FileNotFoundError(dump)
    if not os.path.isfile(csv):
        raise FileNotFoundError(csv)

    atoms = io.read(dump, format="lammps-dump-text", index=":")
    thermo = pd.read_csv(csv)
    positions_z = [i.get_positions()[:, 2] for i in atoms]
    std = np.zeros([len(atoms), num1])
    state = np.zeros([len(atoms), num1], dtype=int)
    density = np.zeros([0, 5])
    cell_z = [i.get_cell()[2][2] for i in atoms]
    volume = [i.get_volume() for i in atoms]

    for i in range(len(atoms)):
        hist, bins = np.histogram(
            positions_z[i],
            bins=np.arange(0, cell_z[i] + cell_z[i] / (num1 * num2), cell_z[i] / (num1 * num2)),
        )
        for j in range(num1):
            std[i][j] = np.std(hist[j * num2 : j * num2 + num2])
            if std[i][j] > ts:
                state[i][j] = 0
            elif std[i][j] < tl:
                state[i][j] = 2
            else:
                state[i][j] = 1
        solid_part = np.zeros(0, dtype=int)
        liquid_part = np.zeros(0, dtype=int)
        interface_part = np.zeros(0, dtype=int)
        solid_density = 0
        liquid_density = 0
        for j in range(num1):
            if state[i][j] == 0:
                solid_part = np.append(solid_part, j)
            if state[i][j] == 1:
                interface_part = np.append(interface_part, j)
            if state[i][j] == 2:
                liquid_part = np.append(liquid_part, j)
        for j in solid_part:
            solid_density += np.sum(hist[j * num2 : j * num2 + num2])
        for j in liquid_part:
            liquid_density += np.sum(hist[j * num2 : j * num2 + num2])
        density_temp = np.zeros([1, 5])
        density_temp[0][0] = i
        if len(solid_part) != 0:
            density_temp[0][1] = solid_density / (volume[i] / num1 * len(solid_part))
            density_temp[0][1] = (3.0 / (4.0 * np.pi * density_temp[0][1])) ** (1.0 / 3.0) / 0.529
        else:
            density_temp[0][1] = None
        if len(liquid_part) != 0:
            density_temp[0][2] = liquid_density / (volume[i] / num1 * len(liquid_part))
            density_temp[0][2] = (3.0 / (4.0 * np.pi * density_temp[0][2])) ** (1.0 / 3.0) / 0.529
        else:
            density_temp[0][2] = None
        density_temp[0][3] = (len(solid_part) + 0.5 * len(interface_part)) / num1
        density_temp[0][4] = (len(liquid_part) + 0.5 * len(interface_part)) / num1
        density = np.r_[density, density_temp]

    np.savetxt(path + f"P{press}T{temp}_state.txt", state, fmt="%d")
    np.savetxt(path + f"P{press}T{temp}_density.txt", density)

    frame2time = args.frame2time
    step2time = args.step2time
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(7, 7))
    axes[0].matshow(state.T, cmap="rainbow", origin="lower")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_ylabel(r"State")
    axes[1].plot(density[:, 0] * frame2time, density[:, 3], label="solid")
    axes[1].plot(density[:, 0] * frame2time, density[:, 4], label="liquid")
    axes[1].set_xlim(0, len(atoms) * frame2time)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel(r"Propotion")
    axes[1].set_xticks([])
    axes[1].legend()
    axes[2].plot(density[:, 0] * frame2time, density[:, 1], label="solid")
    axes[2].plot(density[:, 0] * frame2time, density[:, 2], label="liquid")
    axes[2].set_xlim(0, len(atoms) * frame2time)
    axes[2].set_ylabel(r"$r_s$")
    axes[2].set_xticks([])
    axes[2].legend()
    bar2gpa = 1e-4
    axes[3].plot(thermo["Step"] * step2time, thermo["Pxx"] * bar2gpa, label="pxx")
    axes[3].plot(thermo["Step"] * step2time, thermo["Pyy"] * bar2gpa, label="pyy")
    axes[3].plot(thermo["Step"] * step2time, thermo["Pzz"] * bar2gpa, label="pzz")
    axes[3].set_xlim(0, len(atoms) * frame2time)
    axes[3].set_ylabel(r"$P_d$ (GPa)")
    axes[3].set_xticks([])
    axes[3].legend()
    axes[4].plot(thermo["Step"] * step2time, thermo["Pxy"] * bar2gpa, label="pxy")
    axes[4].plot(thermo["Step"] * step2time, thermo["Pxz"] * bar2gpa, label="pxz")
    axes[4].plot(thermo["Step"] * step2time, thermo["Pyz"] * bar2gpa, label="pyz")
    axes[4].set_xlim(0, len(atoms) * frame2time)
    axes[4].set_ylabel(r"$P_{od}$ (GPa)")
    axes[4].set_xticks([])
    axes[4].legend()
    axes[5].plot(thermo["Step"] * step2time, thermo["Lx"], label="lx")
    axes[5].plot(thermo["Step"] * step2time, thermo["Ly"], label="ly")
    axes[5].plot(thermo["Step"] * step2time, thermo["Lz"], label="lz")
    axes[5].set_xlim(0, len(atoms) * frame2time)
    axes[5].set_ylabel(r"L (A)")
    axes[5].set_xticks([])
    axes[5].legend()
    axes[6].plot(thermo["Step"] * step2time, thermo["Temp"])
    axes[6].set_xlim(0, len(atoms) * frame2time)
    axes[6].set_xlabel(r"Time (ps)")
    axes[6].set_ylabel(r"T (K)")
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    out_fig = path + f"P{press}T{temp}fig.png"
    plt.savefig(out_fig, format="png", dpi=300)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
