import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys, re
sys.path.append("/projects/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen")
sys.path.append("/projects/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack")
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ase import io


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

def data_path(pgpa,tkel,N,phase,case="NPT_mliap_new"):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/{phase}/{case}/{tkel}/analysis/"

def background_gofr_integration():
    background_data = np.loadtxt("/work/nvme/bcqo/shubhanggoswami/PIMD/isentrope_calculation/360atoms/rs1.3935/liquid/NVT/2400/analysis/rs1.3935T2400gofr_info_LAMMPS_liquid.npy")
    bins = background_data[:,0]
    g_r = background_data[:,1]
    mol_idx = bins <= 1.0
    delta_x = np.diff(bins)[mol_idx[:-1]]
    g_r_mol = g_r[mol_idx]
    integr = np.sum(g_r_mol*delta_x)
    return integr


def get_atomic_info_from_PT(P,T,N,phase,case="NPT_mliap"):
    if "ASE" in case:
        data = np.loadtxt(data_path(P,T,N,phase,"NPT_ASE")+f"P{P}T{T}gofr_info_ASE_liquid.npy")
    else:
        data = np.loadtxt(data_path(P,T,N,phase,case)+f"P{P}T{T}gofr_info_LAMMPS_liquid.npy")
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

def get_all_atomicity_data_ASE(all_p):
    molperc_full_set = []
    #glob.glob("p*")
    #all_p = [int(p[1:]) for p in glob.glob("p*")]
    for p in all_p:
        for t in np.array(os.listdir(f"p{p}/liquid/NPT_ASE/"),dtype=int):
            try:
                molperc = get_atomic_info_from_PT_ASE(p,t,'liquid')
                molperc_full_set.append(molperc)
            except Exception as e:
                print(f"Could not get data for P={p} GPa,T={t} K")
                continue
    
    return np.array(molperc_full_set)


def tv_eos(prange,case="NPT_new"):
    N = 360
    Ang2bohr = 1.88973
    molperc_all = get_all_atomicity_data()
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    tkel_for_plotting = []
    rs_for_plotting = []
    tkel_err_for_plotting = []
    rs_err_for_plotting = []
    mperc = []
    for pgpa in prange:
        trange = np.array(os.listdir(f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/{case}/"),dtype=int)
        for tkel in trange:
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
            temp_blocked = reblock(df['Temp'].values,100)
            V_blocked = reblock(df['Volume'].values,100)
            temp_err = np.std(temp_blocked)/np.sqrt(len(temp_blocked))
            v_err = np.std(V_blocked)/np.sqrt(len(V_blocked))
            coeff=(4/3)*np.pi
            rs=(V_blocked/(N*coeff))**(1/3)
            rs = rs*Ang2bohr
            rs_for_plotting.append(np.mean(rs))
            rs_err = np.std(rs)/np.sqrt(len(rs))
            tkel_for_plotting.append(np.mean(tkel))
            tkel_err_for_plotting.append(0)
            rs_err_for_plotting.append(rs_err)
            molperc = get_atomic_info_from_PT(pgpa,tkel,'liquid')
            mperc.append(molperc)
            np.savetxt(f'p{pgpa}/tv_eos_liquid_{N}atoms_{pgpa}GPa.txt',np.c_[tkel_for_plotting,rs_for_plotting,tkel_err_for_plotting,rs_err_for_plotting,mperc],header='T(K)  rs(bohr)  T_err(K)  rs_err(bohr)  mol_percent')
    
    colors = sm.to_rgba(mperc)
    plt.errorbar(rs_for_plotting, tkel_for_plotting, xerr=rs_err_for_plotting, yerr=tkel_err_for_plotting, fmt='none', ecolor='k', capsize=4,alpha=0.5)
    plt.scatter(rs_for_plotting, tkel_for_plotting, c=colors, label=f'{tkel}K',s=80)
    plt.xlabel('r$_s$ (bohr)',fontsize=14)
    plt.ylabel('Temperature (K)',fontsize=14)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    cbar=plt.colorbar(sm,cax=cax)
    cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
    cbar.set_ticklabels(['atomic','molec.'])
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(f'EoS_plots/tv_eos_liquid_360atoms_all.png',dpi=300)
    plt.close()
    np.savetxt(f'EoS_plots/tv_eos_liquid_360atoms_all.txt',np.c_[tkel_for_plotting,rs_for_plotting,tkel_err_for_plotting,rs_err_for_plotting,mperc],header='T(K)  rs(bohr)  T_err(K)  rs_err(bohr)  mol_percent')

def pv_eos(case="NPT_mliap"):
    #prange = np.arange(140,201,10)
    #trange = np.arange(1700,2501,100)
    #LLPT_Tdict={1700:np.arange(190,201,1), 1800:np.arange(180,191,1), 1900:np.arange(160,191,2),2000:np.arange(160,181,2),2100:np.arange(160,181,1),2200:np.arange(150,171,2),2300:np.arange(146,161,2),2400:np.arange(142,160,2),2500:np.arange(130,161,2)}
    #LLPT_Tdict_2={1700:np.arange(190,197,1), 1800:np.arange(181,190,1), 1900:np.arange(175,182,1), 2000:np.arange(160,171,1), 2100:np.arange(156,167,1), 2200:np.arange(153,162,1), 2300:np.arange(144,153,1), 2400:np.arange(138,149,1), 2500:np.arange(133,144,1)}
    #LLPT_Tdict={1900:np.arange(173,181,2), 2000:np.arange(167,173,2), 2100:np.arange(150,160,2), 2200:[145, 147, 155, 157], 2300:np.arange(140,145,2), 2400:np.arange(136,141,2)}
    #LLPT_Tdict = {1700:np.arange(192,196,1), 1800:np.arange(183,187,1), 2000:np.arange(166,171,1), 2100:np.arange(150,171,2), 2300:np.arange(146,151,1), 2500:np.arange(134,140,1)}
    LLPT_Tdict = {
        1700: np.unique(np.concatenate([np.arange(192, 196, 1), np.arange(188, 199, 1)])),
        1800: np.unique(np.concatenate([np.arange(183, 187, 1), np.arange(180, 191, 1)])),
        2000: np.unique(np.concatenate([np.arange(166, 171, 1), np.arange(162, 166, 1), np.arange(171, 175, 1)])),
        2100: np.unique(np.concatenate([np.arange(150, 171, 2), np.arange(157, 168, 2)])),
        #2100: np.arange(150,171,2),
        2300: np.unique(np.concatenate([np.arange(146, 151, 1), np.arange(140, 145, 2), np.arange(152, 157, 2)])),
        2500: np.unique(np.concatenate([np.arange(134, 140, 1), np.arange(128, 135, 2), np.arange(140, 147, 2)])),
        3000: np.arange(96, 128, 2)
    }
    N = 360
    Ang2bohr = 1.88973
    trange = LLPT_Tdict.keys()
    all_p = np.concatenate([LLPT_Tdict[t] for t in trange])
    molperc_all = get_all_atomicity_data(all_p,N,case)
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    trange = LLPT_Tdict.keys()
    for tkel in trange:
        pgpa_for_plotting = []
        rs_for_plotting = []
        pgpa_err_for_plotting = []
        rs_err_for_plotting = []
        mperc = []
        p1_to_add = LLPT_Tdict[tkel]
        v_plotting = []
        v_err_all = []
        # if tkel in LLPT_Tdict_2:
        #     p2_to_add = LLPT_Tdict_2[tkel]
        #     p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
        # else:
        #     print(f"No Key for {tkel} on LLPT_Tdict_2")
        #     p_to_add = p1_to_add
        p_to_add = p1_to_add
        #combined_prange = sorted(list(set(np.concatenate((prange, p1_to_add)))))
        for pgpa in p_to_add:
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
            V_blocked = reblock(df['Volume'].values,100)
            press_err = np.std(press_blocked)/np.sqrt(len(press_blocked))
            v_err = np.std(V_blocked)/np.sqrt(len(V_blocked))
            v_plot = np.mean(V_blocked)
            coeff=(4/3)*np.pi
            rs=(V_blocked/(N*coeff))**(1/3)
            rs = rs*Ang2bohr
            rs_for_plotting.append(np.mean(rs))
            rs_err = np.std(rs)/np.sqrt(len(rs))
            pgpa_for_plotting.append(np.mean(press_blocked))
            pgpa_err_for_plotting.append(press_err)
            rs_err_for_plotting.append(rs_err)
            v_plotting.append(v_plot)
            v_err_all.append(v_err)
            molperc = get_atomic_info_from_PT(pgpa,tkel,N,'liquid',case)
            mperc.append(molperc)
        np.savetxt(f'EoS_plots/eos_liquid_{N}atoms_{tkel}K.txt',np.c_[rs_for_plotting,pgpa_for_plotting,rs_err_for_plotting,pgpa_err_for_plotting,mperc,v_plotting,v_err_all],header='rs(bohr)  Pressure(GPa)  rs_err(bohr)  P_err(GPa)  mol_percent  Volume(A3)  V_err(A3)')

        colors = sm.to_rgba(mperc)
        plt.errorbar(rs_for_plotting, pgpa_for_plotting, xerr=rs_err_for_plotting, yerr=pgpa_err_for_plotting, fmt='none', ecolor='k', capsize=4,alpha=0.5)
        plt.scatter(rs_for_plotting, pgpa_for_plotting, c=colors, label=f'{tkel}K',s=80)
        plt.xlabel('r$_s$ (bohr)',fontsize=14)
        plt.ylabel('Pressure (GPa)',fontsize=14)
        plt.legend(title='Temperature',fontsize=12)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cbar=plt.colorbar(sm,cax=cax)
        cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
        cbar.set_ticklabels(['atomic','molec.'])
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(f'EoS_plots/eos_liquid_{N}atoms_{tkel}K.png',dpi=300)
        plt.close()


def pv_eos_ASE():
    LLPT_Tdict={1700:np.arange(188,199,1), 1800: np.arange(180,191), 1900: np.arange(169,185),2000: np.arange(162,175), 2100: np.arange(150,171,2), 2200: np.arange(146,163,2), 2300: np.arange(140,157,2),
                2400:np.arange(136,153,2),2500:np.arange(128,147,2)}
    N = 360
    Ang2bohr = 1.88973
    trange = LLPT_Tdict.keys()
    all_p = np.concatenate([LLPT_Tdict[t] for t in trange])
    molperc_all = get_all_atomicity_data_ASE(all_p)
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    for tkel in trange:
        pgpa_for_plotting = []
        rs_for_plotting = []
        pgpa_err_for_plotting = []
        rs_err_for_plotting = []
        mperc = []
        p1_to_add = LLPT_Tdict[tkel]
        v_plotting = []
        v_err_all = []
        p_to_add = p1_to_add
        for pgpa in p_to_add:
            try:
                analysis_csv_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/NPT_ASE/{tkel}/analysis/P{pgpa}T{tkel}M29_merged_ase_out.csv"
                df = pd.read_csv(analysis_csv_path)
                df.columns = [re.sub(r'[\s#]','',s) for s in df.columns.values]
            except Exception as e:
                print(f"Could not find data for P={pgpa} GPa,T={tkel} K")
                continue
            analysis_traj_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/NPT_ASE/{tkel}/analysis/P{pgpa}T{tkel}.merged.traj"
            traj = io.read(analysis_traj_path,index=":")
            df = df[df['time']>1]
            if len(df)<500:
                print(f"Not enough data for P={pgpa} GPa,T={tkel} K")
                continue
            Press_val = (df["Pxx"]+df["Pyy"]+df["Pzz"])/3
            press_blocked = reblock(Press_val.values,100)
            vol = np.array([atoms.get_volume() for atoms in traj])
            V_blocked = reblock(vol,100)
            press_err = np.std(press_blocked)/np.sqrt(len(press_blocked))
            v_err = np.std(V_blocked)/np.sqrt(len(V_blocked))
            v_plot = np.mean(V_blocked)
            coeff=(4/3)*np.pi
            rs=(V_blocked/(N*coeff))**(1/3)
            rs = rs*Ang2bohr
            rs_for_plotting.append(np.mean(rs))
            rs_err = np.std(rs)/np.sqrt(len(rs))
            pgpa_for_plotting.append(np.mean(press_blocked))
            pgpa_err_for_plotting.append(press_err)
            rs_err_for_plotting.append(rs_err)
            v_plotting.append(v_plot)
            v_err_all.append(v_err)
            molperc = get_atomic_info_from_PT_ASE(pgpa,tkel,'liquid')
            mperc.append(molperc)
        np.savetxt(f'EoS_plots/ASE/eos_liquid_{N}atoms_{tkel}K.txt',np.c_[rs_for_plotting,pgpa_for_plotting,rs_err_for_plotting,pgpa_err_for_plotting,mperc,v_plotting,v_err_all],header='rs(bohr)  Pressure(GPa)  rs_err(bohr)  P_err(GPa)  mol_percent  Volume(A3)  V_err(A3)')
        colors = sm.to_rgba(mperc)
        plt.errorbar(rs_for_plotting, pgpa_for_plotting, xerr=rs_err_for_plotting, yerr=pgpa_err_for_plotting, fmt='none', ecolor='k', capsize=4,alpha=0.5)
        plt.scatter(rs_for_plotting, pgpa_for_plotting, c=colors, label=f'{tkel}K',s=80)
        plt.xlabel('r$_s$ (bohr)',fontsize=14)
        plt.ylabel('Pressure (GPa)',fontsize=14)
        plt.legend(title='Temperature',fontsize=12)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cbar=plt.colorbar(sm,cax=cax)
        cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
        cbar.set_ticklabels(['atomic','molec.'])
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(f'EoS_plots/ASE/eos_liquid_{N}atoms_{tkel}K.png',dpi=300)
        plt.close()
    
def main():
    N=360
    pv_eos(case="NPT_mliap_new")
    # prange = [int(os.path.basename(name)[1:]) for name in glob.glob(f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p*")]
    # tv_eos(prange)



if __name__ == '__main__':
    main()
