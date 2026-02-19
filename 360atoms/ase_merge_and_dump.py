from ase import io 
import numpy as np
import os,sys
import glob 
import argparse
from natsort import natsorted

def gen_path(P,T,phase,N,case):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{P}/{phase}/{case}/{T}/"

def merge_trajectories(press,temp,phase,N_atoms,case):
    data_path = gen_path(press,temp,phase,N_atoms,case)
    #merge all trajectories under gen path of the form trajectory_NPT_{i}.traj, collect all i in ascending order, read them on through ase io and put them in one and write it under gen_path/analysis
    pattern = os.path.join(data_path,f"trajectory_NPT_*.traj")
    traj_files = natsorted(glob.glob(pattern))
    merged_frames = []
    for f in traj_files:
        frame = io.read(f,index=":")
        merged_frames.extend(frame)

    out_dir = os.path.join(data_path, "analysis")
    #os.makedirs(out_dir, exist_ok=True)

    merged_traj = os.path.join(out_dir, f"P{press}T{temp}.merged.traj")
    # Write merged ASE traj
    io.write(merged_traj, merged_frames,format="traj")
    # Write merged LAMMPS dump text (trajectory)
    #merged_dump = os.path.join(out_dir, f"merged.dump.atom")
    #io.write(merged_dump, merged_frames, format="lammps-dump-text")

def read_log_numeric(path: str) -> np.ndarray:
    """
    Read a log file that has a single header line followed by numeric columns.
    Skips the first line, ignores comment lines starting with '#'.
    Returns shape (nrows, ncols).
    """
    try:
        data = np.loadtxt(path, skiprows=1, comments="#")
    except ValueError as e:
        raise ValueError(f"Failed to parse numeric data from {path}: {e}")

    # If there's only one row, loadtxt returns shape (ncols,), fix it.
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data

def merge_logfiles(press,temp,phase,N_atoms,case):
    data_path = gen_path(press,temp,phase,N_atoms,case)
    pattern = os.path.join(data_path,f"md_*_NPT_*.log")
    files = natsorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    merged = []
    offset =0.0
    for f in files:
        data = read_log_numeric(f)
        if data.shape[1] < 1:
            continue  # nothing usable
        # Shift time column
        data[:, 0] = data[:, 0] + offset
        # Update offset to the last time value (same behavior as your bash script)
        offset = float(data[-1, 0])
        merged.append(data)
    if not merged:
        raise RuntimeError("No data rows were read from the matched log files.")

    merged_arr = np.vstack(merged)
    HEADER = "time, Etot/N, Epot/N, Ekin/N, Temperature , Pxx,Pyy,Pzz,Pyz,Pxz,Pxy"
    np.savetxt(data_path+f"analysis/P{press}T{temp}M29_merged_ase_out.csv",merged_arr,delimiter=',',header=HEADER)

def main():
    parser = argparse.ArgumentParser(description='input P-T to merge the files and dump into lammps text file')
    parser.add_argument('-p','--press',dest='pressure',type=int,help='pressure',required=True)
    parser.add_argument('-t','--temp',dest='temp',type=int,help='temperature',required=True)
    parser.add_argument('-c','--case',dest='case',type=str,help='NPT or NVT or something else',default="NPT")
    parser.add_argument('-f','--phase',dest='phase',type=str,help='solid or liquid',default="liquid")
    parser.add_argument('-n','--natom',dest='natom',type=int,help='number of atoms',required=True)

    args= parser.parse_args()
    press = args.pressure
    temp = args.temp
    case = args.case
    phase= args.phase
    N_atoms = args.natom
    merge_trajectories(press,temp,phase,N_atoms,case)
    merge_logfiles(press,temp,phase,N_atoms,case)
    
if __name__ == "__main__":
    main()