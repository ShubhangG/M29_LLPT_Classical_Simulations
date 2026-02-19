import numpy as np
import pandas as pd
from ase import io
import matplotlib.pyplot as plt 
import re

def reblock(trace, block_size, min_nblock=4, with_sigma=False):
    """ block scalar trace to remove auocorrelation;
    see usage example in reblock_scalar_df
    Args:
        trace (np.array): a trace of scalars, may have multiple columns
        !!!! assuming leading dimension is the number of current blocks.
        block_size (int): size of block in units of current block.
        min_nblock (int,optional): minimum number of blocks needed for
        meaningful statistics, default is 4.
    Returns:
        np.array: re-blocked trace.
    """
    nblock= len(trace)//block_size
    nkeep = nblock*block_size
    if (nblock < min_nblock):
        raise RuntimeError('only %d blocks left after reblock' % nblock)
    # end if
    blocked_trace = trace[:nkeep].reshape(nblock, block_size, *trace.shape[1:])
    ret = np.mean(blocked_trace, axis=1)
    if with_sigma:
        ret = (ret, np.std(blocked_trace, ddof=1, axis=1)/np.sqrt(block_size))
    return ret

def gen_path(P,T,N,case):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{P}/liquid/{case}/{T}/analysis/"


def check_avg_energies_ase_vs_mliap(P,T,N):
    ase_path = gen_path(P,T,N,"NPT_ASE")
    mliap_path = gen_path(P,T,N,"NPT_mliap_new")
    ase_output = pd.read_csv(ase_path+f"P{P}T{T}M29_merged_ase_out.csv")
    ase_output.columns = [re.sub(r'[\s#]','',s) for s in ase_output.columns.values]
    mliap_output = pd.read_csv(mliap_path+f"P{P}T{T}M29_merged_lammps_out.csv")
    Potential_Energy_ASE = reblock(ase_output["Epot/N"].values,1000)
    Potential_Energy_MLIAP = reblock(mliap_output["PotEng"].values,100)/N
    Potential_energy_ase_err = np.std(Potential_Energy_ASE)/np.sqrt(len(Potential_Energy_ASE))
    Potential_energy_mliap_err = np.std(Potential_Energy_MLIAP)/np.sqrt(len(Potential_Energy_MLIAP))
    Potential_energy_ase_avg = np.mean(Potential_Energy_ASE)
    Potential_energy_mliap_avg = np.mean(Potential_Energy_MLIAP)
    print(f"Potential Energy of ASE: {Potential_energy_ase_avg} ± {Potential_energy_ase_err}")
    print(f"Potential Energy of MLIAP: {Potential_energy_mliap_avg} ± {Potential_energy_mliap_err}")
    #Check how many sigmas is the difference between the two energies. 
    difference = Potential_energy_ase_avg - Potential_energy_mliap_avg
    sigma_difference = np.abs(difference/np.sqrt(Potential_energy_ase_err**2 + Potential_energy_mliap_err**2))
    print(f"The difference between the two energies is {difference} eV/atom ± {sigma_difference} sigmas")
    return sigma_difference,Potential_energy_ase_avg,Potential_energy_mliap_avg

def main():
    #Turn this to python
    LLPT_Tdict = {1700:np.arange(192,196,1), 1800:np.arange(183,187,1), 2000:np.arange(166,171,1), 2100:np.arange(150,171,2), 2300:np.arange(146,151,1), 2500:np.arange(134,140,1)}
    #We are going to make a colorplot where y axis is T, x axis is P and the color is sigma_difference and we shall print out any P,T where sigma_difference is greater than 1
    Sigmas = []
    P_vals = []
    T_vals = []
    Potential_energy_ase_avgs = []
    Potential_energy_mliap_avgs = []
    for T in LLPT_Tdict.keys():
        for P in LLPT_Tdict[T]:
            try:
                sigma_difference,Potential_energy_ase_avg,Potential_energy_mliap_avg = check_avg_energies_ase_vs_mliap(P,T,360)
                Sigmas.append(sigma_difference)
                Potential_energy_ase_avgs.append(Potential_energy_ase_avg)
                Potential_energy_mliap_avgs.append(Potential_energy_mliap_avg)
                P_vals.append(P)
                T_vals.append(T)
            except Exception as e:
                print(f"Error for P={P}, T={T}: {e}")
                continue
            if sigma_difference > 2:
                print(f"P={P}, T={T} has a sigma_difference of {sigma_difference}")
            #plt.scatter(P,T,c=sigma_difference,cmap='viridis')
    
    Sigmas = np.array(Sigmas)
    scatter = plt.scatter(P_vals, T_vals, c=Sigmas, cmap='viridis', 
                             vmin=Sigmas.min(), vmax=Sigmas.max())
    plt.colorbar(scatter, label="Sigma Difference")
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Temperature (K)")
    plt.title("Sigma Difference between ASE and MLIAP")
    plt.savefig("MLIAP_vs_ASE_sigma_difference.png")
    plt.close()

    
    plt.hist(Sigmas,bins=30)
    plt.xlabel("Sigma Difference")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sigma Differences")
    plt.savefig("MLIAP_vs_ASE_sigma_difference_histogram.png")
    plt.close()

    Potential_energy_ase_avgs = np.array(Potential_energy_ase_avgs)
    Potential_energy_mliap_avgs = np.array(Potential_energy_mliap_avgs)
    plt.scatter(Potential_energy_ase_avgs,Potential_energy_mliap_avgs)
    #make a 45 degree line
    plt.plot([min(Potential_energy_ase_avgs),max(Potential_energy_ase_avgs)],[min(Potential_energy_ase_avgs),max(Potential_energy_ase_avgs)],'k--')
    plt.xlim(min(Potential_energy_ase_avgs)-0.1,max(Potential_energy_ase_avgs)+0.1)
    plt.ylim(min(Potential_energy_ase_avgs)-0.1,max(Potential_energy_ase_avgs)+0.1)
    plt.xlabel("Potential Energy (ASE) (eV/atom)")
    plt.ylabel("Potential Energy (MLIAP) (eV/atom)")
    plt.title("Potential Energy Comparison between ASE and MLIAP")
    plt.savefig("MLIAP_vs_ASE_potential_energy_comparison.png")
    plt.close()

    #Make histogram of the difference between the two energies
    difference = Potential_energy_ase_avgs - Potential_energy_mliap_avgs
    plt.hist(difference,bins=30)
    plt.xlabel("Difference between ASE and MLIAP (eV/atom)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Difference between ASE and MLIAP")
    plt.savefig("MLIAP_vs_ASE_potentialenergy_difference_histogram.png")
    plt.close()
    return

if __name__ == "__main__":
    main()