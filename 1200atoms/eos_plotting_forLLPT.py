import numpy as np
#from ase.io.trajectory import Trajectory
from ase import io 
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

def data_path(pgpa,tkel,phase,N,case="NPT_mliap"):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/{phase}/{case}/{tkel}/analysis/"

@lru_cache(maxsize=1)
def background_gofr_integration(case="NPT_mliap"):
    if "ASE" in case:
        background_data = np.loadtxt(data_path(140,2400,1200,"liquid","NPT_ASE")+f"P140T2400gofr_info_ASE_liquid.npy")
    else:
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


def tv_eos(prange,case="NPT_mliap"):
    N = 1200
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
        trange = np.array(os.listdir(f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/NPT/"),dtype=int)
        for tkel in trange:
            try:
                analysis_csv_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/NPT/{tkel}/analysis/P{pgpa}T{tkel}M29_merged_lammps_out.csv"
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
    plt.savefig(f'EoS_plots/tv_eos_liquid_1200atoms_all.png',dpi=300)
    plt.close()
    np.savetxt(f'EoS_plots/tv_eos_liquid_1200atoms_all.txt',np.c_[tkel_for_plotting,rs_for_plotting,tkel_err_for_plotting,rs_err_for_plotting,mperc],header='T(K)  rs(bohr)  T_err(K)  rs_err(bohr)  mol_percent')

def pv_eos(case="NPT_mliap"):
    #prange = np.arange(140,201,10)
    #trange = np.arange(1700,2501,100)
    # LLPT_Tdict={1700:[170], 1800:[170,190,192,194], 1900:np.concatenate(([160],np.arange(182,187,2))), 2000:np.arange(164,184,2), 2100:np.arange(156,180,2), 
    #             2200:np.concatenate(([151],np.arange(152,173,10))), 2300:np.concatenate((np.arange(148,158,1),np.arange(162,173,10))), 
    #             2400:np.arange(150,161,10), 2500:np.concatenate((np.arange(131,149,2),np.arange(150,161,10)))}
    # LLPT_Tdict_2={1700:np.arange(190,197,1), 1800:np.arange(181,190,1), 1900:np.arange(175,182,1), 2000:np.arange(160,171,1), 2100:np.arange(156,167,1), 2200:np.arange(153,162,1), 2300:np.arange(144,153,1), 2400:np.arange(138,149,1), 2500:np.arange(133,144,1)}
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
    trange = LLPT_Tdict.keys()
    for tkel in trange:
        pgpa_for_plotting = []
        rs_for_plotting = []
        pgpa_err_for_plotting = []
        rs_err_for_plotting = []
        mperc = []
        v_plotting = []
        v_err_all = []
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
        #combined_prange = sorted(list(set(np.concatenate((prange, p1_to_add)))))
        for pgpa in combined_prange:
            pgpa = np.round(pgpa,1)
            if pgpa.is_integer():
                pgpa = int(pgpa)
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
            molperc = get_atomic_info_from_PT(pgpa,tkel,N,'liquid',case)
            mperc.append(molperc)
            v_plotting.append(v_plot)
            v_err_all.append(v_err)
        np.savetxt(f'EoS_plots/eos_liquid_1200atoms_{tkel}K.txt',np.c_[rs_for_plotting,pgpa_for_plotting,rs_err_for_plotting,pgpa_err_for_plotting,mperc,v_plotting,v_err_all],header='rs(bohr)  Pressure(GPa)  rs_err(bohr)  P_err(GPa)  mol_percent Volume(A3) V_err(A3)')

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
        plt.savefig(f'EoS_plots/eos_liquid_1200atoms_{tkel}K.png',dpi=300)
        plt.close()

def get_all_atomicity_data_ASE(all_p):
    molperc_full_set = []
    #glob.glob("p*")
    #all_p = [int(p[1:]) for p in glob.glob("p*")]
    for p in all_p:
        if len(os.listdir(f"p{p}/liquid/NPT_ASE/"))<1:
            continue 
        for t in np.array(os.listdir(f"p{p}/liquid/NPT_ASE/"),dtype=int):
            try:
                molperc = get_atomic_info_from_PT_ASE(p,t,'liquid')
                molperc_full_set.append(molperc)
            except Exception as e:
                print(f"Could not get data for P={p} GPa,T={t} K")
                continue
    
    return np.array(molperc_full_set)

@lru_cache(maxsize=None)
def get_atomic_info_from_PT_ASE(P,T,phase):
    N=1200
    data = np.loadtxt(data_path(P,T,N,phase,"NPT_ASE")+f"P{P}T{T}gofr_info_ASE_liquid.npy")
    bins = data[:,0]
    g_r = data[:,1]
    mol_idx = bins <= 1.0
    delta_x = np.diff(bins)[mol_idx[:-1]]
    g_r_mol = g_r[mol_idx]
    integr = np.sum(g_r_mol*delta_x)
    #get background to subtract 
    bg_integ = background_gofr_integration()
    return (integr - bg_integ)/bg_integ

def pv_eos_ASE():
    LLPT_Tdict={1700:np.arange(186,201,2), 1800: np.arange(180,191) ,2000: np.arange(160,181), 
                2100: np.arange(156,171,2), 2300: np.arange(140,160,2), 2500:np.arange(130,150,2)}
    LLPT_Tdict2={1700:np.arange(192.5,193.6,1), 1800:np.arange(183,185.1,0.2), 2000: np.arange(167,169.9,0.2),
                2100:np.arange(160.2,164.1,0.2), 2300:np.arange(146.2,149.9,0.2), 2500:np.arange(134.5,137.6,1)}   
    N = 1200
    Ang2bohr = 1.88973
    trange = LLPT_Tdict.keys()
    all_p = np.concatenate([LLPT_Tdict[t] for t in trange])
    molperc_all = get_all_atomicity_data_ASE(all_p)
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    #trange=[2100,2300,2500]
    trange=[2500]
    for tkel in trange:
        pgpa_for_plotting = []
        rs_for_plotting = []
        pgpa_err_for_plotting = []
        rs_err_for_plotting = []
        mperc = []
        p1_to_add = LLPT_Tdict[tkel]
        p2_to_add = LLPT_Tdict2[tkel]
        v_plotting = []
        v_err_all = []
        energy_ev = []
        energy_err = []
        #p_to_add = p1_to_add
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
            analysis_traj_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/NPT_ASE/{tkel}/analysis/P{pgpa}T{tkel}.merged.traj"
            traj = io.read(analysis_traj_path,index="10::10")
            #vol = np.array([atoms.get_volume() for atoms in traj])
            #traj = Trajectory(analysis_traj_path,'r')
            #idx = np.nonzero(df['time'].to_numpy() > 1.0)[0]//100
            #vol = np.fromiter((traj[i].get_volume() for i in idx), dtype=float, count=len(idx))
            vol = np.fromiter((traj[i].get_volume() for i in range(len(traj))),dtype=float,count=len(traj))
            df = df[df['time']>1]
            if len(df)<500:
                print(f"Not enough data for P={pgpa} GPa,T={tkel} K")
                continue
            Press_val = -1.0*(df["Pxx"]+df["Pyy"]+df["Pzz"])/3
            optimal_block_size = 10**4
            press_blocked = reblock(Press_val.values,optimal_block_size)
            E_blocked = reblock(df['Epot/N'].values*N,optimal_block_size)
            V_blocked = reblock(vol,100)
            press_err = np.std(press_blocked)/np.sqrt(len(press_blocked))
            v_err = np.std(V_blocked)/np.sqrt(len(V_blocked))
            e_err = np.std(E_blocked)/np.sqrt(len(E_blocked))
            v_plot = np.mean(V_blocked)
            e_plot = np.mean(E_blocked)
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
            energy_ev.append(e_plot)
            energy_err.append(e_err)
            #molperc = get_atomic_info_from_PT_ASE(pgpa,tkel,'liquid')
            mperc.append(float(get_atomic_info_from_PT_ASE(pgpa, tkel, "liquid")))
        np.savetxt(f'EoS_plots/ASE/eos_liquid_{N}atoms_{tkel}K_updated.txt',np.c_[rs_for_plotting,pgpa_for_plotting,rs_err_for_plotting,pgpa_err_for_plotting,mperc,v_plotting,v_err_all,energy_ev,energy_err],header='rs(bohr)  Pressure(GPa)  rs_err(bohr)  P_err(GPa)  mol_percent  Volume(A3)  V_err(A3) Epot(eV) Epot_err(eV)')
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
        plt.savefig(f'EoS_plots/ASE/eos_liquid_{N}atoms_{tkel}K_updated.png',dpi=300)
        plt.close()
        print(f"Finished {tkel} isotherm!")

def add_E_to_EoS_data():
    LLPT_Tdict={1700:np.arange(186,201,2), 1800: np.arange(180,191) ,2000: np.arange(160,181), 
                2100: np.arange(156,171,2), 2300: np.arange(140,160,2), 2500:np.arange(130,150,2)}
    LLPT_Tdict2={1700:np.arange(192.5,193.6,1), 1800:np.arange(183,185.1,0.2), 2000: np.arange(167,169.9,0.2),
                2100:np.arange(160.2,164.1,0.2), 2300:np.arange(146.2,149.9,0.2), 2500:np.arange(134.5,137.6,1)}  
    N = 1200
    Ang2bohr = 1.88973
    trange = LLPT_Tdict.keys()
    all_p = np.concatenate([LLPT_Tdict[t] for t in trange])
    for tkel in trange:
        eos_path = f"EoS_plots/ASE/eos_liquid_{N}atoms_{tkel}K.txt"
        eos = np.loadtxt(eos_path)

        p1_to_add = LLPT_Tdict[tkel]
        p2_to_add = LLPT_Tdict2[tkel]
        p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
        energy_ev = []
        energy_err = []
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
            E_optimal_block_size = 10**4
            E_reblocked = reblock(df['Epot/N'].values*N,E_optimal_block_size)
            energy_ev.append(np.mean(E_reblocked))
            E_se = np.std(E_reblocked)/np.sqrt(len(E_reblocked))
            energy_err.append(E_se)
        pv_eos = np.loadtxt()


    


def main():
    N=1200
    #prange = [float(os.path.basename(name)[1:]) for name in glob.glob(f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p*")]
    #tv_eos(prange)
    pv_eos()
    #pv_eos_ASE()



if __name__ == '__main__':
    main()
