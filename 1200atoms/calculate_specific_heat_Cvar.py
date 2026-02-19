import numpy as np
import matplotlib.pyplot as plt
import sys,os,glob,re
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
    block_size_all = range(1, len(trace)//10, 10)
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

def data_path(pgpa,tkel,phase,N=1200,case="NPT_mliap"):
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

    

def get_all_atomicity_data_ASE(all_p,N):
    molperc_full_set = []
    #glob.glob("p*")
    #all_p = [int(p[1:]) for p in glob.glob("p*")]
    for p in all_p:
        if len(os.listdir(f"p{p}/liquid/NPT_ASE/"))<1:
            continue 
        for t in np.array(os.listdir(f"p{p}/liquid/NPT_ASE/"),dtype=int):
            try:
                molperc = get_atomic_info_from_PT_ASE(p,t,N,'liquid')
                molperc_full_set.append(molperc)
            except Exception as e:
                print(f"Could not get data for P={p} GPa,T={t} K")
                continue
    
    return np.array(molperc_full_set)

@lru_cache(maxsize=None)
def get_atomic_info_from_PT_ASE(P,T,N,phase):
    data = np.loadtxt(data_path(P,T,phase,N,"NPT_ASE")+f"P{P}T{T}gofr_info_ASE_liquid.npy")
    bins = data[:,0]
    g_r = data[:,1]
    mol_idx = bins <= 1.0
    delta_x = np.diff(bins)[mol_idx[:-1]]
    g_r_mol = g_r[mol_idx]
    integr = np.sum(g_r_mol*delta_x)
    #get background to subtract 
    bg_integ = background_gofr_integration()
    return (integr - bg_integ)/bg_integ

def cv_lammps(case="NPT_mliap"):
    #prange = np.arange(140,201,10)
    #trange = np.arange(1700,2501,100)
    # LLPT_Tdict={1700:[170], 1800:[170,190,192,194], 1900:np.concatenate(([160],np.arange(182,187,2))), 2000:np.arange(164,184,2), 2100:np.arange(156,180,2), 
    #             2200:np.concatenate(([151],np.arange(152,173,10))), 2300:np.concatenate((np.arange(148,158,1),np.arange(162,173,10))), 
    #             2400:np.arange(150,161,10), 2500:np.concatenate((np.arange(131,149,2),np.arange(150,161,10)))}
    # LLPT_Tdict_2={1700:np.arange(190,197,1), 1800:np.arange(181,190,1), 1900:np.arange(175,182,1), 2000:np.arange(160,171,1), 2100:np.arange(156,167,1), 2200:np.arange(154,162,1), 2300:np.arange(144,153,1), 2400:np.arange(138,149,1), 2500:np.arange(133,144,1)}
    # #LLPT_Tdict=([1700]=$(seq 193 1 196) [1800]=$(seq 184 1 184) [1900]=$(seq 176 1 177) [2000]=$(seq 169 1 171) [2100]=$(seq 161 1 163) [2200]=$(seq 154 1 157) [2300]=$(seq 148 1 150) [2400]=$(seq 142 1 145) [2500]=$(seq 137 1 140))
    # LLPT_Tdict_3={1700:np.arange(191.5,195,1),1800:np.arange(183 ,185.1,0.2),1900:np.arange(176.2,178,0.2),2000:np.arange(168,170,0.2),2100:np.arange(160.2,165,0.2), 
    #               2200:np.arange(154.2,157,0.2), 2300:np.arange(148.2,150,0.2),2400: np.arange(142.5,145.6,0.5)}
    LLPT_Tdict = {
    1700: np.arange(192, 194.5, 0.5),      # seq 192 0.5 194
    1800: np.arange(183, 185.2, 0.2),      # seq 183 0.2 185
    2000: np.arange(167.4, 169.0, 0.2),    # seq 167.4 0.2 168.8
    2100: np.arange(160, 162.2, 0.2),      # seq 160 0.2 162
    2300: np.arange(146.2, 149.5, 0.2),    # seq 146.2 0.2 149.4
    2500: np.arange(134, 138.5, 0.5),      # seq 134 0.5 138
    3000: np.arange(96, 128, 2)             # seq 96 2 126
    }

    LLPT_Tdict_2 = {
    1700: np.array([186, 188, 190, 196, 198]),
    1800: np.array([180, 182, 186, 188]),
    2000: np.concatenate([np.arange(162, 168, 2), np.arange(170, 176, 2)]),  # seq 162 2 166; seq 170 2 174
    2100: np.array([156, 158, 163, 164, 166]),
    2300: np.array([142, 144, 150, 152]),
    2500: np.array([130, 132, 140, 142])
}

    N = 1200
    Ang2bohr = 1.88973
    trange = LLPT_Tdict.keys()
    all_p = np.concatenate([LLPT_Tdict[t] for t in trange])
    molperc_all = get_all_atomicity_data(all_p,N,case)
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    max_llpt_P = []
    max_cv = []
    max_cv_err = []
    trange = LLPT_Tdict.keys()
    for tkel in trange:
        pgpa_for_plotting = []
        cv_for_plotting = []
        cv_err_for_plotting = []
        pgpa_err_for_plotting = []
        mperc = []
        if tkel ==3000:
            p1_to_add = LLPT_Tdict[tkel]
            combined_prange = p1_to_add
        else:
            p1_to_add = LLPT_Tdict[tkel]
            p2_to_add = LLPT_Tdict_2[tkel]
            p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
            combined_prange = p_to_add
        # p1_to_add = LLPT_Tdict[tkel]
        # p2_to_add = LLPT_Tdict_2[tkel]
        # if tkel==2500:
        #     p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
        # else:
        #     p3_to_add = LLPT_Tdict_3[tkel]
        #     p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add,p3_to_add)))
        #p_to_add = LLPT_Tdict[tkel]
        for pgpa in combined_prange:
            pgpa = np.round(pgpa,1)
            if pgpa.is_integer():
                pgpa=int(pgpa)
            try:
                analysis_csv_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/{case}/{tkel}/analysis/P{pgpa}T{tkel}M29_merged_lammps_out.csv"
                df = pd.read_csv(analysis_csv_path)
            except Exception as e:
                print(f"Could not find data for P={pgpa} GPa,T={tkel} K")
                continue
            df = df[df['Step']>1000]
            if len(df)<500:
                print(f"Not enough data for P={pgpa} GPa,T={tkel} K")
                continue
            press_blocked = reblock(df['Press'].values,100)/10**4
            E_optimal_block_size, E_optimal_mean, E_optimal_SE = block_analysis(df['PotEng'].values,os.path.dirname(analysis_csv_path),'Energy')
            #E_reblocked, E_SE = reblock(df['PotEng'].values,E_optimal_block_size,with_sigma=True)
            #E_var = (E_SE*E_optimal_block_size**0.5)**2
            E_var = np.var(df['PotEng'].values)
            cv = E_var/N
            cv_to_block = (df['PotEng'].values - np.mean(df['PotEng'].values))**2/N
            cv_blocked = reblock(cv_to_block,E_optimal_block_size)
            cv_err = np.std(cv_blocked)/np.sqrt(len(cv_blocked))
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
        plt.savefig(f'Cv_plots/Cv_liquid_1200atoms_{tkel}K.png',dpi=300)
        plt.close()
        #get maximum cv and corresponding pressure
        max_cv_idx = np.argmax(cv_for_plotting)
        max_llpt_P.append(pgpa_for_plotting[max_cv_idx])
        max_cv.append(cv_for_plotting[max_cv_idx])
        max_cv_err.append(cv_err_for_plotting[max_cv_idx])
        #save cv for plotting
        np.savetxt(f'Cv_plots/Cv_liquid_1200atoms_{tkel}K_data.txt',np.array([pgpa_for_plotting,cv_for_plotting,cv_err_for_plotting]).T,header='Pressure(GPa) Cv Cv_err')

    #plotting max cv pressure vs temperature; add colorbar for max_cv
    norm = Normalize(vmin=min(max_cv),vmax=max(max_cv))
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    colors = sm.to_rgba(max_cv)
    print(max_llpt_P,trange)
    #save maxllpt, P and trange
    np.savetxt('Cv_plots/Cv_liquid_1200atoms_maxcv_data.txt', np.column_stack([list(trange), max_llpt_P, max_cv, max_cv_err]),
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
    plt.savefig(f'Cv_plots/Cv_liquid_1200atoms_max_pressure_vs_temp.png',dpi=300)
    plt.close()

    plt.errorbar(trange,max_cv,yerr=max_cv_err,fmt='o-')
    plt.ylabel('max(Cv) ',fontsize=14)
    plt.xlabel('Temperature (K)',fontsize=14)
    plt.savefig(f'Cv_plots/Cv_liquid_1200atoms_temp_vs_maxcv.png',dpi=300)
    plt.close()


def cv_ASE():
    LLPT_Tdict={1700:np.arange(186,201,2), 1800: np.arange(180,191) ,2000: np.arange(160,181), 
                2100: np.arange(156,171,2), 2300: np.arange(140,160,2), 2500:np.arange(130,150,2)}
    LLPT_Tdict2={1700:np.arange(192.5,193.6,1), 1800:np.arange(183,185.1,0.2), 2000: np.arange(167,169.9,0.2),
                2100:np.arange(160.2,164.1,0.2), 2300:np.arange(146.2,149.9,0.2), 2500:np.arange(134.5,137.6,1)}

    N = 1200
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
    for tkel in trange:
        pgpa_for_plotting = []
        cv_for_plotting = []
        cv_err_for_plotting = []
        pgpa_err_for_plotting = []
        mperc = []
        p1_to_add = LLPT_Tdict[tkel]
        p2_to_add = LLPT_Tdict2[tkel]
        p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
        for pgpa in p_to_add:
            pgpa = np.round(pgpa,1)
            if pgpa.is_integer():
                pgpa=int(pgpa)
            try:
                analysis_csv_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/NPT_ASE/{tkel}/analysis/P{pgpa}T{tkel}M29_merged_ase_out.csv"
                df = pd.read_csv(analysis_csv_path)
                df.columns = [re.sub(r'[\s#]','',s) for s in df.columns.values]
            except Exception as e:
                print(f"Could not find data for P={pgpa} GPa,T={tkel} K")
                continue
            df = df[df['time']>1]
            if len(df)<5000:
                print(f"Not enough data for P={pgpa} GPa,T={tkel} K")
                continue
            Press_val = -1.0*(df["Pxx"]+df["Pyy"]+df["Pzz"])/3
            press_blocked = reblock(Press_val.values,10**4)
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
            molperc = get_atomic_info_from_PT(pgpa,tkel,N,'liquid','NPT_ASE')
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
    cv_lammps("NPT_mliap")
    #cv_ASE()


if __name__ == "__main__":
    main()