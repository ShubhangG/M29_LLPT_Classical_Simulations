import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from ase import io

def gen_data_path(press,temp,N_atoms,case):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N_atoms}atoms/p{press}/liquid/{case}/{temp}/analysis/"

def plot_lammps_density(press,temp,N_atoms,case):
    data_path = gen_data_path(press,temp,N_atoms,case)
    data = pd.read_csv(data_path+f"P{press}T{temp}M29_merged_lammps_out.csv")
    data = data[data['Step']>1000]
    time = data['Time'].values
    pgpa = data['Press'].values/10**4
    vol = data['Volume'].values
    rho = N_atoms/vol
    plt.plot(time,rho)
    plt.xlabel('Time (ps)')
    plt.ylabel('Density ($A^{-3}$)')
    plt.title(f"P={press} GPa, T={temp} K")
    plt.savefig(data_path+f"P{press}T{temp}_density_vs_time.png",dpi=200)
    
def plot_ASE_density(press,temp,N_atoms,case):
    data_path = gen_data_path(press,temp,N_atoms,case)
    traj = io.read(data_path+f"P{press}T{temp}.merged.traj", index="10:")
    vol = np.array([atoms.get_volume() for atoms in traj])
    rho = N_atoms/vol
    time = (np.arange(0,len(vol))+10)*0.005
    plt.plot(time,rho)
    plt.xlabel('Time (ps)')
    plt.ylabel('Density ($A^{-3}$)')
    plt.title(f"P={press} GPa, T={temp} K")
    plt.savefig(data_path+f"P{press}T{temp}_density_vs_time.png",dpi=200)
    


def main():
    parser = ArgumentParser(description='input P-T to plot the liquid trajectories')
    parser.add_argument('-p','--press',dest='pressure',type=float,help='pressure',required=True)
    parser.add_argument('-t','--temp',dest='temp',type=int,help='temperature',required=True)
    parser.add_argument('-c','--case',dest='case',type=str,help='NPT or NVT or something else',default="NPT")
    parser.add_argument('-n','--natom',dest='natom',type=int,help="Number of atoms",required=True)

    args= parser.parse_args()
    press = args.pressure
    if press.is_integer():
        press = int(press)
    temp = args.temp
    case = args.case
    N_atoms = args.natom
    if "ASE" in case:
        plot_ASE_density(press,temp,N_atoms,case)
    else:
        plot_lammps_density(press,temp,N_atoms,case)

    return

if __name__ == "__main__":
    main()