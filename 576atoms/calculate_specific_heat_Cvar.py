import numpy as np
import matplotlib.pyplot as plt
import sys,os,glob, re
sys.path.append("/projects/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen")
sys.path.append("/projects/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack")
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import lru_cache

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


def block_analysis(trace,savefile_folder,quantity='Energy'):
    Rx_over_blocks = [] 
    mean_over_blocks = []
    full_var = np.var(trace)
    SE_over_blocks = []
    block_size_all = range(1, int(np.sqrt(len(trace))), 10)
    for block_size in block_size_all:
        try:
            blocked_trace = reblock(trace, block_size, min_nblock=4)
            mean_value = np.mean(blocked_trace)
            Rx = block_size*np.var(blocked_trace)/full_var
            SE = np.std(blocked_trace)/np.sqrt(len(blocked_trace))
            Rx_over_blocks.append(Rx)
            mean_over_blocks.append(mean_value)
            SE_over_blocks.append(SE)
        except Exception as e:
            continue
    
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(block_size_all, Rx_over_blocks)
    ax[0].set_xlabel('Block Size')
    ax[0].set_xscale('log')
    ax[0].set_ylabel('Rx')
    ax[0].set_title('Block Analysis')
    ax[1].plot(block_size_all, mean_over_blocks)
    ax[1].set_xlabel('Block Size')
    ax[1].set_ylabel('Mean Value')
    ax[1].set_title('Mean Value over Block Size')
    plt.tight_layout()
    plt.savefig(f'{savefile_folder}/{quantity}_block_analysis.png')
    plt.close()
    max_idx = np.argmax(Rx_over_blocks)
    optimal_block_size = block_size_all[max_idx]
    optimal_SE = SE_over_blocks[max_idx]
    optimal_mean = mean_over_blocks[max_idx]
    return optimal_block_size, optimal_mean, optimal_SE

def data_path(pgpa,tkel,phase,N=576,case="NPT"):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/{phase}/{case}/{tkel}/analysis/"

@lru_cache(maxsize=1)
def background_gofr_integration():
    background_data = np.loadtxt("/work/nvme/bcqo/shubhanggoswami/PIMD/isentrope_calculation/360atoms/rs1.3935/liquid/NVT/2400/analysis/rs1.3935T2400gofr_info_LAMMPS_liquid.npy")
    bins = background_data[:,0]
    g_r = background_data[:,1]
    mol_idx = bins <= 1.0
    delta_x = np.diff(bins)[mol_idx[:-1]]
    g_r_mol = g_r[mol_idx]
    integr = np.sum(g_r_mol*delta_x)
    return integr

@lru_cache(maxsize=None)
def get_atomic_info_from_PT(P,T,N,phase,case="NPT_mliap"):
    if "ASE" in case:
        data = np.loadtxt(data_path(P,T,phase,N,"NPT_ASE")+f"P{P}T{T}gofr_info_ASE_liquid.npy")
    else:
        data = np.loadtxt(data_path(P,T,phase,N,case)+f"P{P}T{T}gofr_info_LAMMPS_liquid.npy")
    bins = data[:,0]
    g_r = data[:,1]
    mol_idx = bins <= 1.0
    delta_x = np.diff(bins)[mol_idx[:-1]]
    g_r_mol = g_r[mol_idx]
    integr = np.sum(g_r_mol*delta_x)
    #get background to subtract 
    bg_integ = background_gofr_integration()
    return (integr - bg_integ)/bg_integ

def get_all_atomicity_data(all_p,N,case="NPT_mliap"):
    molperc_full_set = []
    #all_p = [int(p[1:]) for p in glob.glob("p*")]
    for p in all_p:
        p=np.round(p,1)
        if p.is_integer():
            p=int(p)
        if len(os.listdir(f"p{p}/liquid/{case}/"))<1:
            continue 
        for t in np.array(os.listdir(f"p{p}/liquid/{case}/"),dtype=int):
            try:
                molperc = get_atomic_info_from_PT(p,t,N,'liquid',case)
                molperc_full_set.append(molperc)
            except Exception as e:
                print(f"Could not get data for P={p} GPa,T={t} K")
                continue
    
    return np.array(molperc_full_set)

def cv_lammps(case="NPT_mliap"):
    prange = np.arange(150,201,10)
    trange = np.arange(1700,2501,100)
    # LLPT_Tdict={1700:np.arange(191,200,2),1800:np.arange(181,190,2), 1900:np.arange(171,179,2), 2000:np.arange(161,180,2), 2100:np.arange(161,170,2),
    #             2200:np.arange(151,160,2),2300:np.arange(141,150,2), 2400:np.arange(141,150,2),2500:np.arange(150,161,10)}
    # LLPT_Tdict_2={1700:np.arange(191,196,1),1800:np.arange(181,190,1), 1900:np.arange(171,180,1), 2000:[164,166,168,169,170,171], 2100:np.arange(156,162,1),
    #             2200:np.arange(151,159,1),2300:np.arange(148,159,1), 2400:np.arange(138,149,1),2500:np.arange(131,150,2)}
    LLPT_Tdict = {
    1700: np.arange(191, 195, 1),      # seq 191 1 194
    1800: np.arange(183, 185.5, 0.5),  # seq 183 0.5 185
    2000: np.arange(166, 171, 1),      # seq 166 1 170
    2100: np.arange(159, 163.5, 0.5),  # seq 159 0.5 163
    2300: np.arange(146, 151, 1),      # seq 146 1 150
    2500: np.arange(134, 139, 1),       # seq 134 1 138
    3000: np.arange(96, 128, 2)
    }

    LLPT_Tdict_2 = {
    1700: np.arange(186, 200, 2),      # seq 186 2 198
    1800: np.arange(180, 189, 1),      # seq 180 1 188
    2000: np.arange(160, 174, 2),      # seq 160 2 174
    2100: np.arange(156, 170, 2),      # seq 156 2 168
    2300: np.arange(140, 158, 2),      # seq 140 2 156
    2500: np.arange(130, 144, 2),      # seq 130 2 142
}
    
    N = 576
    Ang2bohr = 1.88973
    trange = LLPT_Tdict.keys()
    all_p = np.concatenate([LLPT_Tdict[t] for t in trange])
    #all_p = np.concatenate([[int(p) if np.round(p,1).is_integer() else np.round(p,1) for p in LLPT_Tdict[t]] for t in trange])
    #print(all_p)
    molperc_all = get_all_atomicity_data(all_p,N,case)
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    max_llpt_P = []
    max_cv = []
    max_cv_err =[]
    for tkel in trange:
        pgpa_for_plotting = []
        cv_for_plotting = []
        cv_err_for_plotting = []
        pgpa_err_for_plotting = []
        mperc = []
        if tkel ==3000:
            p1_to_add = LLPT_Tdict[tkel]
            #combined_prange = sorted(list(set(np.concatenate((prange, p2_to_add)))))
            combined_prange = p1_to_add
        else:
            p1_to_add = LLPT_Tdict[tkel]
            p2_to_add = LLPT_Tdict_2[tkel]
            p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
            #combined_prange = sorted(list(set(np.concatenate((prange, p_to_add)))))
            combined_prange = p_to_add
        for pgpa in combined_prange:
            pgpa = np.round(pgpa,1)
            if pgpa.is_integer():
                pgpa=int(pgpa)
            try:
                analysis_csv_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/576atoms/p{pgpa}/liquid/{case}/{tkel}/analysis/P{pgpa}T{tkel}M29_merged_lammps_out.csv"
                df = pd.read_csv(analysis_csv_path)
            except Exception as e:
                print(f"Could not find data for P={pgpa} GPa,T={tkel} K")
                continue
            press_blocked = reblock(df['Press'].values,100)/10**4
            E_optimal_block_size, E_optimal_mean, E_optimal_SE = block_analysis(df['PotEng'].values,os.path.dirname(analysis_csv_path),'Energy')
            #E_reblocked, E_SE = reblock(df['PotEng'].values,E_optimal_block_size,with_sigma=True)
            #E_var = (E_SE*E_optimal_block_size**0.5)**2
            E_var = np.var(df['PotEng'].values)
            #print(len(E_SE),len(E_var))
            cv = E_var/N
            cv_to_block = (df['PotEng'].values - np.mean(df['PotEng'].values))**2/N
            cv_blocked = reblock(cv_to_block,E_optimal_block_size)
            cv_err = np.std(cv_blocked)/np.sqrt(len(cv_blocked))
            if np.isnan(cv):
                print(f"NaN Cv for P={pgpa} GPa,T={tkel} K")
                continue
            press_err = np.std(press_blocked)/np.sqrt(len(press_blocked))
            pgpa_for_plotting.append(np.mean(press_blocked))
            pgpa_err_for_plotting.append(press_err)
            cv_for_plotting.append(cv)
            cv_err_for_plotting.append(cv_err)
            molperc = get_atomic_info_from_PT(pgpa,tkel,N,'liquid',case)
            mperc.append(molperc)
        colors = sm.to_rgba(mperc)
        plt.errorbar(pgpa_for_plotting, cv_for_plotting, xerr=pgpa_err_for_plotting, yerr=cv_err_for_plotting, fmt='none', ecolor='k', capsize=4,alpha=0.5)
        plt.scatter(pgpa_for_plotting, cv_for_plotting, c=colors, label=f'{tkel}K',s=80)
        plt.ylabel('Variance of PotEng/N',fontsize=14)
        plt.xlabel('Pressure (GPa)',fontsize=14)
        plt.legend(title='Temperature',fontsize=12)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cbar=plt.colorbar(sm,cax=cax)
        cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
        cbar.set_ticklabels(['atomic','molec.'])
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(f'Cv_plots/Cv_liquid_576atoms_{tkel}K.png',dpi=300)
        plt.close()
        #get maximum cv and corresponding pressure
        max_cv_idx = np.argmax(cv_for_plotting)
        max_llpt_P.append(pgpa_for_plotting[max_cv_idx])
        max_cv.append(cv_for_plotting[max_cv_idx])
        max_cv_err.append(cv_err_for_plotting[max_cv_idx])
        np.savetxt(f'Cv_plots/Cv_liquid_576atoms_{tkel}K_data.txt',np.array([pgpa_for_plotting,cv_for_plotting,cv_err_for_plotting]).T,header='Pressure(GPa) Cv Cv_err')


    #plotting max cv pressure vs temperature; add colorbar for max_cv
    norm = Normalize(vmin=min(max_cv),vmax=max(max_cv))
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    colors = sm.to_rgba(max_cv)
    print(max_llpt_P,trange)
    np.savetxt('Cv_plots/Cv_liquid_576atoms_maxcv_data.txt', np.column_stack([list(trange), max_llpt_P, max_cv, max_cv_err]),
        header='Temperature(K) Pressure(GPa) max(Cv) max(Cv)_err')
    plt.plot(max_llpt_P,trange,color='k',alpha=0.5)
    plt.scatter(max_llpt_P,trange,c=colors,s=100)
    plt.xlabel('Pressure (GPa)',fontsize=14)
    plt.ylabel('Temperature (K)',fontsize=14)
    #ADD colorbar for max_cv
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    cbar=plt.colorbar(sm,cax=cax)
    cbar.set_ticks([np.min(max_cv),np.max(max_cv)])
    cbar.set_ticklabels([f'min Cv={min(max_cv):.2f}',f'max Cv={max(max_cv):.2f}'])
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(f'Cv_plots/Cv_liquid_576atoms_max_pressure_vs_temp.png',dpi=300)

    plt.close()
    plt.errorbar(trange,max_cv,yerr=max_cv_err,fmt='o-')
    plt.ylabel('max(Cv) ',fontsize=14)
    plt.xlabel('Temperature (K)',fontsize=14)
    plt.savefig(f'Cv_plots/Cv_liquid_576atoms_temp_vs_maxcv.png',dpi=300)
    plt.close()


def cv_ASE():
    LLPT_Tdict={1700:np.arange(186,201,2), 1800: np.arange(180,191),2000: np.arange(160,181), 
                2100: np.arange(156,171,2), 2300: np.arange(140,161,2),2500:np.arange(130,151,2)}
    #LLPT_Tdict2={1700:np.arange(188,196.1,0.5), 1800:np.arange(182,185.1,0.2), 2000: np.arange(165.5,171.6,0.5),
    #             2100:np.arange(158,164.1,0.5), 2300:np.arange(143,152.1,0.5), 2500:np.arange(136.5,141.1)}
    LLPT_Tdict2={1700:np.arange(188,197,1), 2100:np.arange(158,165,1), 2300:np.arange(143,153,1), 2500:np.arange(136,142)}

    N = 576
    Ang2bohr = 1.88973
    trange = LLPT_Tdict.keys()
    all_p = np.concatenate([LLPT_Tdict[t] for t in trange])
    molperc_all = get_all_atomicity_data(all_p,N,case="NPT_ASE")
    max_llpt_P = []
    max_cv = []
    max_cv_err = []
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    #trange = [2500]
    for tkel in trange:
        pgpa_for_plotting = []
        cv_for_plotting = []
        cv_err_for_plotting = []
        pgpa_err_for_plotting = []
        mperc = []
        p1_to_add = LLPT_Tdict[tkel]
        if tkel in LLPT_Tdict2:
            p2_to_add = LLPT_Tdict2[tkel]
            p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
        else:
            p_to_add = p1_to_add
        for pgpa in p_to_add:
            # pgpa = np.round(pgpa,1)
            # if pgpa.is_integer():
            #     pgpa=int(pgpa)
            try:
                analysis_csv_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/NPT_ASE/{tkel}/analysis/P{pgpa}T{tkel}M29_merged_ase_out.csv"
                df = pd.read_csv(analysis_csv_path)
                df.columns = [re.sub(r'[\s#]','',s) for s in df.columns.values]
            except Exception as e:
                print(f"Could not find data for P={pgpa} GPa,T={tkel} K")
                continue
            df = df[df['time']>1]
            if len(df)<500:
                print(f"Not enough data for P={pgpa} GPa,T={tkel} K")
                continue
            Press_val = -1.0*(df["Pxx"]+df["Pyy"]+df["Pzz"])/3
            press_blocked = reblock(Press_val.values,10**3)
            #E_optimal_block_size, E_optimal_mean, E_optimal_SE = block_analysis(df['Epot/N'].values,os.path.dirname(analysis_csv_path),'Energy')
            E_optimal_block_size = 10**3
            #E_reblocked, E_SE = reblock(df['Epot/N'].values*N,E_optimal_block_size,with_sigma=True)
            #E_var = (E_SE*E_optimal_block_size**0.5)**2
            E_var = np.var(df['Epot/N'].values*N)
            cv = E_var/N
            cv_to_block = (df['Epot/N'].values*N - np.mean(df['Epot/N'].values*N))**2/N
            cv_blocked = reblock(cv_to_block,E_optimal_block_size)
            cv_err = np.std(cv_blocked)/np.sqrt(len(cv_blocked))
            press_err = np.std(press_blocked)/np.sqrt(len(press_blocked))
            pgpa_for_plotting.append(np.mean(press_blocked))
            pgpa_err_for_plotting.append(press_err)
            cv_for_plotting.append(cv)
            cv_err_for_plotting.append(cv_err)
            #molperc = get_atomic_info_from_PT_ASE(pgpa,tkel,N,'liquid')
            molperc = float(get_atomic_info_from_PT(pgpa, tkel, N, "liquid","NPT_ASE"))
            mperc.append(molperc)
        colors = sm.to_rgba(mperc)
        plt.errorbar(pgpa_for_plotting, cv_for_plotting, xerr=pgpa_err_for_plotting, yerr=cv_err_for_plotting, fmt='none', ecolor='k', capsize=4,alpha=0.5)
        plt.scatter(pgpa_for_plotting, cv_for_plotting, c=colors, label=f'{tkel}K',s=80)
        plt.ylabel('Variance of PotEng/N',fontsize=14)
        plt.xlabel('Pressure (GPa)',fontsize=14)
        plt.legend(title='Temperature',fontsize=12)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cbar=plt.colorbar(sm,cax=cax)
        cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
        cbar.set_ticklabels(['atomic','molec.'])
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(f'Cv_plots/ASE/Cv_liquid_{N}atoms_{tkel}K.png',dpi=300)
        plt.close()
        #get maximum cv and corresponding pressure
        max_cv_idx = np.argmax(cv_for_plotting)
        max_llpt_P.append(pgpa_for_plotting[max_cv_idx])
        max_cv.append(cv_for_plotting[max_cv_idx])
        max_cv_err.append(cv_err_for_plotting[max_cv_idx])
        #save cv for plotting
        np.savetxt(f'Cv_plots/ASE/Cv_liquid_{N}atoms_{tkel}K_data.txt',np.array([pgpa_for_plotting,cv_for_plotting,cv_err_for_plotting]).T,header='Pressure(GPa) Cv Cv_err')
        print(f"Isotherm {tkel} K finished! ")

    #plotting max cv pressure vs temperature; add colorbar for max_cv
    norm = Normalize(vmin=min(max_cv),vmax=max(max_cv))
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    colors = sm.to_rgba(max_cv)
    print(max_llpt_P,trange)
    #save maxllpt, P and trange
    np.savetxt(f'Cv_plots/ASE/Cv_liquid_{N}atoms_maxcv_data.txt', np.column_stack([list(trange), max_llpt_P, max_cv, max_cv_err]),
        header='Temperature(K) Pressure(GPa) max(Cv) max(Cv)_err')

    plt.plot(max_llpt_P,trange,color='k',alpha=0.5)
    plt.scatter(max_llpt_P,trange,c=colors,s=100)
    plt.xlabel('Pressure (GPa)',fontsize=14)
    plt.ylabel('Temperature (K)',fontsize=14)
    #ADD colorbar for max_cv
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    cbar=plt.colorbar(sm,cax=cax)
    cbar.set_ticks([np.min(max_cv),np.max(max_cv)])
    cbar.set_ticklabels([f'min Cv={min(max_cv):.2f}',f'max Cv={max(max_cv):.2f}'])
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(f'Cv_plots/ASE/Cv_liquid_{N}atoms_max_pressure_vs_temp.png',dpi=300)
    plt.close()

    plt.errorbar(trange,max_cv,yerr=max_cv_err,fmt='o-')
    plt.ylabel('max(Cv) ',fontsize=14)
    plt.xlabel('Temperature (K)',fontsize=14)
    plt.savefig(f'Cv_plots/ASE/Cv_liquid_{N}atoms_temp_vs_maxcv.png',dpi=300)
    plt.close()


def main():
    cv_lammps(case="NPT_mliap")
    #cv_ASE()

if __name__ == "__main__":
    main()