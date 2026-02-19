import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def get_path(press,tkel,case):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/{case}/{tkel}/analysis"

def main():
    parser = ArgumentParser(description='input P-T to plot the liquid trajectories')
    parser.add_argument('-p','--press',dest='pressure',type=int,help='pressure',required=True)
    parser.add_argument('-t','--temp',dest='temp',type=int,help='temperature',required=True)
    parser.add_argument('-c','--case',dest='case',type=str,help='NPT or NVT or something else',default="NPT")

    args= parser.parse_args()
    press = args.pressure
    temp = args.temp
    case = args.case
    N_atoms = 360
    data_path = get_path(press,temp,case)
    md_data = np.loadtxt(data_path+f"/P{press}T{temp}_NPT_merged.dat",skiprows=1)
    df = pd.DataFrame(md_data,columns=['time','Etot/N','Epot/N','Ekin/N','T(K)','Pxx','Pyy','Pzz','Pxy','Pxz','Pyz'])
    df.drop_duplicates(inplace=True)
    df = df[df['time']>1]

    plot_energy_pressure(data_path,df,press,temp,case)


def plot_energy_pressure(data_path,df,press,temp,case):
    Press = (df['Pxx'] + df['Pyy'] + df['Pzz']) / 3.0
    time = df['time']
    energy = df['Etot/N']
    fig, (ax_e, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), dpi=150)

    ax_e.plot(time, energy, color="tab:blue")
    ax_e.set_ylabel("Etot/N")
    ax_e.set_title(f"Energy trace P={press} GPa, T={temp} K")
    ax_e.grid(True, linewidth=0.3, alpha=0.6)

    ax_p.plot(time, Press, color="tab:red")
    ax_p.set_xlabel("Time")
    ax_p.set_ylabel("Pressure (GPa)")
    ax_p.set_title("Average pressure")
    ax_p.grid(True, linewidth=0.3, alpha=0.6)

    fig.tight_layout()
    plt.savefig(data_path+f"/P{press}T{temp}_{case}_energy_pressure_trace.png")

if __name__ == "__main__":
    main()
    # md_data = np.loadtxt("/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p162/liquid/NPT_ASE_3/2100/md_2100K_162.0GPa_NPT_0.log",skiprows=1)
    # df = pd.DataFrame(md_data,columns=['time','Etot/N','Epot/N','Ekin/N','T(K)','Pxx','Pyy','Pzz','Pxy','Pxz','Pyz'])
    # df.drop_duplicates(inplace=True)
    # df = df[df['time']>1]

    # plot_energy_pressure("/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p162/liquid/NPT_ASE_3/2100/",df,162,2100,"NPT")

    
