#!/usr/bin/env python3
"""Extract last frame of a LAMMPS dump to a LAMMPS data file."""
from __future__ import annotations

import argparse
import sys

from ase.io import read

sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/mysharelib")
from mytool import myio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--index", type=int, default=-1)
    args = parser.parse_args()

    atoms = read(args.input, index=args.index, format="lammps-dump-text")
    atoms.set_pbc(True)
    myio.atoms2data(atoms, args.output, ["H"])
    print(f"Wrote {args.output}: {len(atoms)} atoms, cell={atoms.cell.lengths()}")


if __name__ == "__main__":
    main()
