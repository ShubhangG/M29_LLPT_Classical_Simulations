#!/usr/bin/env python3
"""Merge lammps.out*.txt thermo for twophase NPT runs into analysis/ CSV."""
from natsort import natsorted
import os
import argparse
import numpy as np

ROOT = "/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations"


def gen_data_path(P, T, phase="twophase", case="NPT_mliap"):
    return f"{ROOT}/p{P}/{phase}/{case}/{T}/"


def merge_and_dump_lammps(P, T, phase="twophase", case="NPT_mliap"):
    run_dir = gen_data_path(P, T, phase, case)
    analysis = os.path.join(run_dir, "analysis")
    os.makedirs(analysis, exist_ok=True)
    all_data = natsorted(os.listdir(run_dir))
    simout_filelist = [x for x in all_data if "lammps.out" in x]
    if "lammps.out.init.txt" in simout_filelist:
        simout_filelist.remove("lammps.out.init.txt")
        simout_filelist.insert(0, "lammps.out.init.txt")

    out_csv = os.path.join(analysis, f"P{P}T{T}M29_merged_lammps_out.csv")
    FO = open(out_csv, "w")
    sim_out_array = []
    header_written = False
    for file_name in simout_filelist:
        with open(os.path.join(run_dir, file_name)) as fin:
            for row in fin.read().split("\n"):
                columns = row.split()
                if len(columns) != 22:
                    continue
                if columns[0] == "Step":
                    if not header_written:
                        FO.write(",".join(columns) + "\n")
                        header_written = True
                    continue
                FO.write(",".join(columns) + "\n")
                sim_out_array.append(list(map(float, columns)))
    FO.close()
    np.save(os.path.join(analysis, f"P{P}T{T}M29_merged_lammps_out"), np.array(sim_out_array))
    print(f"Wrote {out_csv} ({len(sim_out_array)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--press", type=float, required=True)
    parser.add_argument("-t", "--temp", type=int, required=True)
    parser.add_argument("-c", "--case", default="NPT_mliap")
    parser.add_argument("-f", "--phase", default="twophase")
    args = parser.parse_args()
    press = int(args.press) if float(args.press).is_integer() else args.press
    merge_and_dump_lammps(press, args.temp, args.phase, args.case)


if __name__ == "__main__":
    main()
