import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Avoid OpenBLAS thread limit error on HPC
import numpy as np
import matplotlib.pyplot as plt
import sys,re
from ase.io.trajectory import Trajectory
#from ase import io
import glob 
sys.path.append("/projects/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen")
sys.path.append("/projects/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack")
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import lru_cache
from sklearn.mixture import GaussianMixture


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

def tv_eos(prange):
    N = 576
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
    plt.savefig(f'EoS_plots/tv_eos_liquid_{N}atoms_all.png',dpi=300)
    plt.close()
    np.savetxt(f'EoS_plots/tv_eos_liquid_{N}atoms_all.txt',np.c_[tkel_for_plotting,rs_for_plotting,tkel_err_for_plotting,rs_err_for_plotting,mperc],header='T(K)  rs(bohr)  T_err(K)  rs_err(bohr)  mol_percent')


def pv_eos(case="NPT_mliap"):
    # prange = np.arange(140,201,10)
    # trange = np.arange(1700,2501,100)
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
    2500: np.arange(134, 139, 1)       # seq 134 1 138
    }

    LLPT_Tdict_2 = {
    1700: np.arange(186, 200, 2),      # seq 186 2 198
    1800: np.arange(180, 189, 1),      # seq 180 1 188
    2000: np.arange(160, 174, 2),
    2100: np.arange(156, 170, 2),      # seq 156 2 168
    2300: np.arange(140, 158, 2),      # seq 140 2 156
    2500: np.arange(130, 144, 2),      # seq 130 2 142
    3000: np.arange(96, 128, 2)        # seq 96 2 126
    }
    N = 576
    Ang2bohr = 1.88973
    trange = LLPT_Tdict_2.keys()
    all_p = np.concatenate([LLPT_Tdict_2[t] for t in trange])
    #all_p = np.concatenate([[int(p) if np.round(p,1).is_integer() else np.round(p,1) for p in LLPT_Tdict[t]] for t in trange])
    molperc_all = get_all_atomicity_data(all_p,N,case)
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    for tkel in trange:
        pgpa_for_plotting = []
        rs_for_plotting = []
        pgpa_err_for_plotting = []
        rs_err_for_plotting = []
        mperc = []
        v_plotting = []
        v_err_all = []
        if tkel == 3000:
            p2_to_add = LLPT_Tdict_2[tkel]
            #combined_prange = sorted(list(set(np.concatenate((prange, p2_to_add)))))
            combined_prange = p2_to_add
        else:
            p1_to_add = LLPT_Tdict[tkel]
            p2_to_add = LLPT_Tdict_2[tkel]
            p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
            #combined_prange = sorted(list(set(np.concatenate((prange, p_to_add)))))
            combined_prange = p_to_add
        #combined_prange = LLPT_Tdict[tkel]
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
            df = df[df['Step']>2000]
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

        np.savetxt(f'EoS_plots/eos_liquid_576atoms_{tkel}K.txt',np.c_[rs_for_plotting,pgpa_for_plotting,rs_err_for_plotting,pgpa_err_for_plotting,mperc,v_plotting,v_err_all],header='rs(bohr)  Pressure(GPa)  rs_err(bohr)  P_err(GPa)  mol_percent Volume(A3) V_err(A3)')

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
        plt.savefig(f'EoS_plots/eos_liquid_576atoms_{tkel}K.png',dpi=300)
        plt.close()


def pf_eos(case="NPT_mliap"):
    """Pressure (x) vs c_total_force_sq/N (y), colored by mol_percent."""
    LLPT_Tdict = {
        1700: np.arange(191, 195, 1),
        1800: np.arange(183, 185.5, 0.5),
        2000: np.arange(166, 171, 1),
        2100: np.arange(159, 163.5, 0.5),
        2300: np.arange(146, 151, 1),
        2500: np.arange(134, 139, 1)
    }
    LLPT_Tdict_2 = {
        1700: np.arange(186, 200, 2),
        1800: np.arange(180, 189, 1),
        2000: np.arange(160, 175, 2),
        2100: np.arange(156, 170, 2),
        2300: np.arange(140, 158, 2),
        2500: np.arange(130, 144, 2),
        3000: np.arange(96, 128, 2)
    }

    fitting_ppoints_dict = {
        1700: (4, 2),
        1800: (3, 3),
        2000: (3, 2),
        2100: (3, 2),
        2300: (3, 3),
        2500: (2, 2),
        3000: (4, 4),
    }

    N = 576
    trange = LLPT_Tdict_2.keys()
    trange=[1700,1800]
    all_p = np.concatenate([LLPT_Tdict_2[t] for t in trange])
    molperc_all = get_all_atomicity_data(all_p, N, case)
    norm = Normalize(vmin=min(molperc_all), vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    delta_F2_all = {}
    delta_V_all = {}
    for tkel in trange:
        pgpa_for_plotting = []
        pf_for_plotting = []
        pgpa_err_for_plotting = []
        pf_err_for_plotting = []
        mperc = []
        volume_for_plotting = []
        volume_err_for_plotting = []
        csv_paths = []
        if tkel == 3000:
            combined_prange = LLPT_Tdict_2[tkel]
        else:
            p1_to_add = LLPT_Tdict[tkel]
            p2_to_add = LLPT_Tdict_2[tkel]
            combined_prange = np.unique(np.concatenate((p1_to_add, p2_to_add)))
        for pgpa in np.sort(combined_prange):
            pgpa = np.round(pgpa, 1)
            if pgpa.is_integer():
                pgpa = int(pgpa)
            try:
                analysis_csv_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/{case}/{tkel}/analysis/P{pgpa}T{tkel}M29_merged_lammps_out.csv"
                df = pd.read_csv(analysis_csv_path)
            except Exception as e:
                print(f"Could not find data for P={pgpa} GPa,T={tkel} K")
                continue
            if 'c_total_force_sq' not in df.columns:
                print(f"c_total_force_sq not in {analysis_csv_path}")
                continue
            df = df[df['Step'] > 2000]
            if len(df) < 500:
                print(f"Not enough data for P={pgpa} GPa,T={tkel} K")
                continue
            press_blocked = reblock(df['Press'].values, 100) / 10**4
            volume_blocked = reblock(df['Volume'].values, 100) / N
            force_sq_blocked = reblock(df['c_total_force_sq'].values, 100) / N
            press_err = np.std(press_blocked) / np.sqrt(len(press_blocked))
            pf_err = np.std(force_sq_blocked) / np.sqrt(len(force_sq_blocked))
            volume_err = np.std(volume_blocked) / np.sqrt(len(volume_blocked))
            pgpa_for_plotting.append(np.mean(press_blocked))
            pgpa_err_for_plotting.append(press_err)
            pf_for_plotting.append(np.mean(force_sq_blocked))
            pf_err_for_plotting.append(pf_err)
            volume_for_plotting.append(np.mean(volume_blocked))
            volume_err_for_plotting.append(volume_err)
            molperc = get_atomic_info_from_PT(pgpa, tkel, N, 'liquid', case)
            mperc.append(molperc)
            csv_paths.append(analysis_csv_path)
        np.savetxt(f'EoS_plots/pf_eos_liquid_{N}atoms_{tkel}K.txt',
                   np.c_[pgpa_for_plotting, pf_for_plotting, pgpa_err_for_plotting, pf_err_for_plotting, mperc, volume_for_plotting, volume_err_for_plotting],
                   header='Pressure(GPa)  c_total_force_sq/N  P_err(GPa)  pf_err  mol_percent  Volume(A3) Volume_err(A3)')
        colors = sm.to_rgba(mperc)

        left_fit, right_fit = fit_pressure_vs_func_values(pgpa_for_plotting, pf_for_plotting, fitting_ppoints_dict[tkel])
        left_fit_equation = np.poly1d(left_fit)
        right_fit_equation = np.poly1d(right_fit)
        left_fit_volume,right_fit_volume = fit_pressure_vs_func_values(pgpa_for_plotting, volume_for_plotting, fitting_ppoints_dict[tkel])
        left_fit_volume_equation = np.poly1d(left_fit_volume)
        right_fit_volume_equation = np.poly1d(right_fit_volume)
        #Read the transition point from eos_liquid_576atoms_{tkel}K_fit_params_fromvolume.txt
        params = np.loadtxt(f'EoS_plots/eos_liquid_576atoms_{tkel}K_fit_params_fromvolume.txt')
        Pt = params[2]
        #Now get the fit value for both at the transition point.
        left_fit_value = left_fit_equation(Pt)
        right_fit_value = right_fit_equation(Pt)
        # Bimodal decomposition of transition-region points
        left_pts, right_pts = fitting_ppoints_dict[tkel]
        n_pts = len(pgpa_for_plotting)
        #the transition points start from the left pts +1 amd then end at the right pts -1.
        trans_start, trans_end = left_pts+1, n_pts - right_pts
        pgpa_mol, pf_mol, pf_mol_err, pgpa_atom, pf_atom, pf_atom_err = [], [], [], [], [], []
        pgpa_mol_err, pgpa_atom_err = [], []
        # bimodal_mean_dict = {
        #     1700: {192: (4.5, 6}
        # }
        for i in range(trans_start, trans_end):
            try:
                df_raw = pd.read_csv(csv_paths[i])
                df_raw = df_raw[df_raw['Step'] > 2000]
                pf_raw = (df_raw['c_total_force_sq'].values / N).astype(float)
                mean_low, mean_high, err_low, err_high = fit_bimodal_means(pf_raw)
                if mean_low is not None:
                    pf_mol.append(mean_low)
                    pf_mol_err.append(err_low if err_low is not None else 0.0)
                    pgpa_mol.append(pgpa_for_plotting[i])
                    pgpa_mol_err.append(pgpa_err_for_plotting[i] if i < len(pgpa_err_for_plotting) else 0.0)
                if mean_high is not None:
                    pf_atom.append(mean_high)
                    pf_atom_err.append(err_high if err_high is not None else 0.0)
                    pgpa_atom.append(pgpa_for_plotting[i])
                    pgpa_atom_err.append(pgpa_err_for_plotting[i] if i < len(pgpa_err_for_plotting) else 0.0)
            except Exception as e:
                pass
            #Also show the histogram and the fits to  the two bimodal distributions.
            # plt.hist(pf_raw, bins=30, alpha=0.5)
            # if mean_low is not None:
            #     plt.axvline(x=mean_low, color='red', ls='--',label='Atomic Fit')
            # if mean_high is not None:
            #     plt.axvline(x=mean_high, color='green', ls='--', label='Molecular Fit')
            # plt.legend()
            # plt.xlabel('F2/N (eV2/A2)', fontsize=14)
            # plt.ylabel('Count', fontsize=14)
            # plt.title(f'{tkel}K', fontsize=14)
            # plt.savefig(f'{os.path.dirname(csv_paths[i])}/histogram_pf_{N}atoms_{tkel}K_whistogramfit.png', dpi=300)
            # plt.close()
        if len(pgpa_mol) > 0 and len(pgpa_atom) > 0:
            plt.errorbar(pgpa_mol, pf_mol, xerr=pgpa_mol_err, yerr=pf_mol_err, fmt='s', markersize=10, c='darkgreen',
                        markeredgecolor='black', markeredgewidth=1.5, capsize=4, label='Bimodal: atomic', zorder=5)
            plt.errorbar(pgpa_atom, pf_atom, xerr=pgpa_atom_err, yerr=pf_atom_err, fmt='^', markersize=10, c='darkblue',
                        markeredgecolor='black', markeredgewidth=1.5, capsize=4, label='Bimodal: molecular', zorder=5)
            np.savetxt(f'EoS_plots/bimodal_pf_{N}atoms_{tkel}K.txt',
                       np.c_[pgpa_mol, pf_mol, pf_mol_err, pf_atom, pf_atom_err],
                       header='P(GPa)  F2/N_molecular  F2/N_molecular_err  F2/N_atomic  F2/N_atomic_err')
        plt.errorbar(pgpa_for_plotting, pf_for_plotting, xerr=pgpa_err_for_plotting, yerr=pf_err_for_plotting,
            fmt='none', ecolor='k', capsize=4, alpha=0.5)
        plt.scatter(pgpa_for_plotting, pf_for_plotting, c=colors, label=f'{tkel}K', s=80)
        plt.legend(title='Temperature', fontsize=12)
        #Create a np arange points between the minimum and Pt and then the Pt and the maximum.
        pgpa_points = np.linspace(min(pgpa_for_plotting), Pt, 20)
        pgpa_points_2 = np.linspace(Pt, max(pgpa_for_plotting), 20)
        left_fit_points = left_fit_equation(pgpa_points)
        right_fit_points = right_fit_equation(pgpa_points_2)
        #Make a plot of the data and the two fits.
        plt.plot(pgpa_points, left_fit_points, label='Molecular Fit', color='red')
        plt.plot(pgpa_points_2, right_fit_points, label='Atomic Fit', color='red')
        #Add the transition point to the plot as a vertical line.
        plt.axvline(x=Pt, color='black', linestyle='--')
        plt.xlabel('P (GPa)', fontsize=14)
        plt.ylabel(r'$\langle F^2 \rangle$ / N (eV$^2$/A$^2$)', fontsize=14)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cbar=plt.colorbar(sm,cax=cax)
        cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
        cbar.set_ticklabels(['atomic','molec.'])
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(f'EoS_plots/pf_eos_liquid_{N}atoms_{tkel}K_wbimodalfit.png', dpi=300)
        plt.close()
        
        #Make plots with Volume vs Pressure
        plt.errorbar(pgpa_for_plotting, volume_for_plotting, xerr=pgpa_err_for_plotting, yerr=volume_err_for_plotting,
                     fmt='none', ecolor='k', capsize=4, alpha=0.5)
        plt.scatter(pgpa_for_plotting, volume_for_plotting, c=colors, label=f'{tkel}K', s=80)
        plt.legend(title='Temperature', fontsize=12)
        plt.xlabel('P (GPa)', fontsize=14)
        plt.ylabel('V (A3) / N', fontsize=14)
        left_fit_volume_value = left_fit_volume_equation(Pt)
        right_fit_volume_value = right_fit_volume_equation(Pt)
        pgpa_points = np.linspace(min(pgpa_for_plotting), Pt, 20)
        pgpa_points_2 = np.linspace(Pt, max(pgpa_for_plotting), 20)
        left_fit_volume_points = left_fit_volume_equation(pgpa_points)
        right_fit_volume_points = right_fit_volume_equation(pgpa_points_2)
        #Add the transition point to the plot as a vertical line.
        plt.plot(pgpa_points, left_fit_volume_points, label='Molecular Fit', color='red')
        plt.plot(pgpa_points_2, right_fit_volume_points, label='Atomic Fit', color='red')
        plt.axvline(x=Pt, color='black', linestyle='--')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cbar=plt.colorbar(sm,cax=cax)
        cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
        cbar.set_ticklabels(['atomic','molec.'])
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(f'EoS_plots/volume_eos_liquid_{N}atoms_{tkel}K_wfit.png', dpi=300)
        plt.close()
        delta_F2_all[tkel] = left_fit_value - right_fit_value
        delta_V_all[tkel] = left_fit_volume_value - right_fit_volume_value
    
    #instead of saving two dictionaries make one np save txt with columns Temperature(K)  delta_F2  delta_V
    np.savetxt(f'EoS_plots/delta_F2_and_V_all.txt',np.c_[list(delta_F2_all.keys()),list(delta_F2_all.values()),list(delta_V_all.values())],header='Temperature(K)  delta_F2  delta_V')
    return delta_F2_all, delta_V_all




def pf_eos_split_by_means(case="NPT_mliap"):
    """Pressure (x) vs c_total_force_sq/N (y), colored by mol_percent."""
    LLPT_Tdict = {
        1700: np.arange(191, 195, 1),
        1800: np.arange(183, 185.5, 0.5),
        2000: np.arange(166, 171, 1),
        2100: np.arange(159, 163.5, 0.5),
        2300: np.arange(146, 151, 1),
        2500: np.arange(134, 139, 1)
    }
    LLPT_Tdict_2 = {
        1700: np.arange(186, 200, 2),
        1800: np.arange(180, 189, 1),
        2000: np.arange(160, 175, 2),
        2100: np.arange(156, 170, 2),
        2300: np.arange(140, 158, 2),
        2500: np.arange(130, 144, 2),
        3000: np.arange(96, 128, 2)
    }

    fitting_ppoints_dict = {
        1700: (4, 2),
        1800: (3, 3),
        2000: (3, 2),
        2100: (3, 2),
        2300: (3, 3),
        2500: (2, 2),
        3000: (4, 4),
    }

    N = 576
    trange = LLPT_Tdict_2.keys()
    trange=[1700,1800]
    all_p = np.concatenate([LLPT_Tdict_2[t] for t in trange])
    molperc_all = get_all_atomicity_data(all_p, N, case)
    norm = Normalize(vmin=min(molperc_all), vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    delta_F2_all = {}
    delta_V_all = {}
    for tkel in trange:
        pgpa_for_plotting = []
        pf_for_plotting = []
        pgpa_err_for_plotting = []
        pf_err_for_plotting = []
        mperc = []
        volume_for_plotting = []
        volume_err_for_plotting = []
        csv_paths = []
        if tkel == 3000:
            combined_prange = LLPT_Tdict_2[tkel]
        else:
            p1_to_add = LLPT_Tdict[tkel]
            p2_to_add = LLPT_Tdict_2[tkel]
            combined_prange = np.unique(np.concatenate((p1_to_add, p2_to_add)))
        for pgpa in np.sort(combined_prange):
            pgpa = np.round(pgpa, 1)
            if pgpa.is_integer():
                pgpa = int(pgpa)
            try:
                analysis_csv_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/{case}/{tkel}/analysis/P{pgpa}T{tkel}M29_merged_lammps_out.csv"
                df = pd.read_csv(analysis_csv_path)
            except Exception as e:
                print(f"Could not find data for P={pgpa} GPa,T={tkel} K")
                continue
            if 'c_total_force_sq' not in df.columns:
                print(f"c_total_force_sq not in {analysis_csv_path}")
                continue
            df = df[df['Step'] > 2000]
            if len(df) < 500:
                print(f"Not enough data for P={pgpa} GPa,T={tkel} K")
                continue
            press_blocked = reblock(df['Press'].values, 100) / 10**4
            volume_blocked = reblock(df['Volume'].values, 100) / N
            force_sq_blocked = reblock(df['c_total_force_sq'].values, 100) / N
            press_err = np.std(press_blocked) / np.sqrt(len(press_blocked))
            pf_err = np.std(force_sq_blocked) / np.sqrt(len(force_sq_blocked))
            volume_err = np.std(volume_blocked) / np.sqrt(len(volume_blocked))
            pgpa_for_plotting.append(np.mean(press_blocked))
            pgpa_err_for_plotting.append(press_err)
            pf_for_plotting.append(np.mean(force_sq_blocked))
            pf_err_for_plotting.append(pf_err)
            volume_for_plotting.append(np.mean(volume_blocked))
            volume_err_for_plotting.append(volume_err)
            molperc = get_atomic_info_from_PT(pgpa, tkel, N, 'liquid', case)
            mperc.append(molperc)
            csv_paths.append(analysis_csv_path)
        np.savetxt(f'EoS_plots/pf_eos_liquid_{N}atoms_{tkel}K.txt',
                   np.c_[pgpa_for_plotting, pf_for_plotting, pgpa_err_for_plotting, pf_err_for_plotting, mperc, volume_for_plotting, volume_err_for_plotting],
                   header='Pressure(GPa)  c_total_force_sq/N  P_err(GPa)  pf_err  mol_percent  Volume(A3) Volume_err(A3)')
        colors = sm.to_rgba(mperc)

        left_fit, right_fit = fit_pressure_vs_func_values(pgpa_for_plotting, pf_for_plotting, fitting_ppoints_dict[tkel])
        left_fit_equation = np.poly1d(left_fit)
        right_fit_equation = np.poly1d(right_fit)
        left_fit_volume,right_fit_volume = fit_pressure_vs_func_values(pgpa_for_plotting, volume_for_plotting, fitting_ppoints_dict[tkel])
        left_fit_volume_equation = np.poly1d(left_fit_volume)
        right_fit_volume_equation = np.poly1d(right_fit_volume)
        #Read the transition point from eos_liquid_576atoms_{tkel}K_fit_params_fromvolume.txt
        params = np.loadtxt(f'EoS_plots/eos_liquid_576atoms_{tkel}K_fit_params_fromvolume.txt')
        Pt = params[2]
        #Now get the fit value for both at the transition point.
        left_fit_value = left_fit_equation(Pt)
        right_fit_value = right_fit_equation(Pt)
        # Bimodal decomposition of transition-region points
        left_pts, right_pts = fitting_ppoints_dict[tkel]
        n_pts = len(pgpa_for_plotting)
        #the transition points start from the left pts +1 amd then end at the right pts -1.

        trans_start, trans_end = left_pts, n_pts - right_pts
        pgpa_mol, pf_mol, pf_mol_err, pgpa_atom, pf_atom, pf_atom_err = [], [], [], [], [], []
        pgpa_mol_err, pgpa_atom_err = [], []
        # bimodal_mean_dict = {
        #     1700: {192: (4.5, 6}
        # }
        for i in range(trans_start, trans_end):
            splitting_force_sq_dict = {
                1700: {
                    'atomic': 5,
                    'molecular': 5.25
                },
                1800: {
                    'atomic': 5.6,
                    'molecular': 5.4
                }
            }
            try:
                df_raw = pd.read_csv(csv_paths[i])
                df_raw = df_raw[df_raw['Step'] > 2000]
                pf_raw = (df_raw['c_total_force_sq'].values / N).astype(float)
                atomic_f2 = pf_raw[pf_raw <= splitting_force_sq_dict[tkel]['atomic']]
                molecular_f2 = pf_raw[pf_raw >= splitting_force_sq_dict[tkel]['molecular']]
                blocked_atomic_f2 = reblock(atomic_f2, 250)
                blocked_molecular_f2 = reblock(molecular_f2, 250)
                atomic_f2_mean = np.mean(blocked_atomic_f2)
                molecular_f2_mean = np.mean(blocked_molecular_f2)
                atomic_f2_err = np.std(blocked_atomic_f2) / np.sqrt(len(blocked_atomic_f2))
                molecular_f2_err = np.std(blocked_molecular_f2) / np.sqrt(len(blocked_molecular_f2))
                pf_mol.append(molecular_f2_mean)
                pf_mol_err.append(molecular_f2_err)
                pf_atom.append(atomic_f2_mean)
                pf_atom_err.append(atomic_f2_err)
                pgpa_mol.append(pgpa_for_plotting[i])
                pgpa_mol_err.append(pgpa_err_for_plotting[i])
                pgpa_atom.append(pgpa_for_plotting[i])
                pgpa_atom_err.append(pgpa_err_for_plotting[i])
            except Exception as e:
                pass
        plt.errorbar(pgpa_mol, pf_mol, xerr=pgpa_mol_err, yerr=pf_mol_err,
            fmt='^', markersize=10, c='darkblue', markeredgecolor='black', markeredgewidth=1.5, capsize=4, label='Molecular', zorder=5)
        plt.errorbar(pgpa_atom, pf_atom, xerr=pgpa_atom_err, yerr=pf_atom_err,
            fmt='s', markersize=10, c='darkgreen', markeredgecolor='black', markeredgewidth=1.5, capsize=4, label='Atomic', zorder=5)
        plt.scatter(pgpa_for_plotting , pf_for_plotting, c=colors, s=80)
        plt.legend(title='phase', fontsize=12, loc='best',frameon=False)
        #Create a np arange points between the minimum and Pt and then the Pt and the maximum.
        pgpa_points = np.linspace(min(pgpa_for_plotting), Pt, 20)
        pgpa_points_2 = np.linspace(Pt, max(pgpa_for_plotting), 20)
        left_fit_points = left_fit_equation(pgpa_points)
        right_fit_points = right_fit_equation(pgpa_points_2)
        #Make a plot of the data and the two fits.
        plt.plot(pgpa_points, left_fit_points, label='Molecular Fit', color='red')
        plt.plot(pgpa_points_2, right_fit_points, label='Atomic Fit', color='red')
        #Add the transition point to the plot as a vertical line.
        plt.axvline(x=Pt, color='black', linestyle='--')
        plt.xlabel('P (GPa)', fontsize=14)
        plt.ylabel(r'$\langle F^2 \rangle$ / N (eV$^2$/A$^2$)', fontsize=14)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cbar=plt.colorbar(sm,cax=cax)
        cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
        cbar.set_ticklabels(['atomic','molec.'])
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(f'EoS_plots/pf_eos_liquid_{N}atoms_{tkel}K_wmeansplit.png', dpi=300)
        plt.close()
        
        #Make plots with Volume vs Pressure
        plt.errorbar(pgpa_for_plotting, volume_for_plotting, xerr=pgpa_err_for_plotting, yerr=volume_err_for_plotting,
                     fmt='none', ecolor='k', capsize=4, alpha=0.5)
        plt.scatter(pgpa_for_plotting, volume_for_plotting, c=colors, label=f'{tkel}K', s=80)
        plt.legend(title='Temperature', fontsize=12)
        plt.xlabel('P (GPa)', fontsize=14)
        plt.ylabel('V (A3) / N', fontsize=14)
        left_fit_volume_value = left_fit_volume_equation(Pt)
        right_fit_volume_value = right_fit_volume_equation(Pt)
        pgpa_points = np.linspace(min(pgpa_for_plotting), Pt, 20)
        pgpa_points_2 = np.linspace(Pt, max(pgpa_for_plotting), 20)
        left_fit_volume_points = left_fit_volume_equation(pgpa_points)
        right_fit_volume_points = right_fit_volume_equation(pgpa_points_2)
        #Add the transition point to the plot as a vertical line.
        plt.plot(pgpa_points, left_fit_volume_points, label='Molecular Fit', color='red')
        plt.plot(pgpa_points_2, right_fit_volume_points, label='Atomic Fit', color='red')
        plt.axvline(x=Pt, color='black', linestyle='--')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cbar=plt.colorbar(sm,cax=cax)
        cbar.set_ticks([np.min(molperc_all),np.max(molperc_all)])
        cbar.set_ticklabels(['atomic','molec.'])
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(f'EoS_plots/volume_eos_liquid_{N}atoms_{tkel}K_wfit.png', dpi=300)
        plt.close()
        delta_F2_all[tkel] = left_fit_value - right_fit_value
        delta_V_all[tkel] = left_fit_volume_value - right_fit_volume_value
    
    #instead of saving two dictionaries make one np save txt with columns Temperature(K)  delta_F2  delta_V
    np.savetxt(f'EoS_plots/delta_F2_and_V_all.txt',np.c_[list(delta_F2_all.keys()),list(delta_F2_all.values()),list(delta_V_all.values())],header='Temperature(K)  delta_F2  delta_V')
    return delta_F2_all, delta_V_all

def fit_bimodal_means(values, min_points=100):
    """Fit a 2-component GMM to fluctuating values; return (mean_low, mean_high, err_low, err_high) for molecular/atomic phases.
    err_low, err_high are standard errors of the mean for each Gaussian: SE = std/sqrt(n) computed from the sample of points assigned to that component."""
    if len(values) < min_points:
        return None, None, None, None
    values = np.asarray(values).reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
        gmm.fit(values)
        means = gmm.means_.flatten()
        idx_low = np.argmin(means)
        idx_high = np.argmax(means)
        mean_low = float(means[idx_low])
        mean_high = float(means[idx_high])
        # Assign points to components and compute sample std (and SE) for each Gaussian
        labels = gmm.predict(values)
        vals_low = values[labels == idx_low].flatten()
        vals_high = values[labels == idx_high].flatten()
        n_low, n_high = len(vals_low), len(vals_high)
        std_low = np.std(vals_low, ddof=1) if n_low > 1 else 0.0
        std_high = np.std(vals_high, ddof=1) if n_high > 1 else 0.0
        err_low = float(std_low / np.sqrt(n_low)) if n_low > 0 else 0.0
        err_high = float(std_high / np.sqrt(n_high)) if n_high > 0 else 0.0
        return mean_low, mean_high, err_low, err_high
    except Exception:
        return None, None, None, None


#Create a function that fits the pressure vs c_total_force but only those away from the inflection point (say first 3 and last 3 points) to 2 separate straight lines. THe atomic straight line and the molecular straight line. Ignore the points in the transtition region.
# points is a tuple that has how many points of each side to use for the two straight lines. Pt is the transition point.
def fit_pressure_vs_func_values(pgpa_for_plotting, func_values_for_plotting,points):
    left_points = points[0]
    right_points = points[1]
    left_pgpa = pgpa_for_plotting[:left_points]
    left_func_values = func_values_for_plotting[:left_points]
    right_pgpa = pgpa_for_plotting[-right_points:]
    right_func_values = func_values_for_plotting[-right_points:]
    left_fit = np.polyfit(left_pgpa, left_func_values, 1)
    right_fit = np.polyfit(right_pgpa, right_func_values, 1)

    
    return left_fit, right_fit

def align_df_to_traj(df_full, traj_path, time_col="time"):
    """
    Returns:
      traj (Trajectory)
      df_traj (DataFrame-like): downsampled df with length ~= len(traj),
                                aligned so row i corresponds to traj[i].
    """
    traj = Trajectory(traj_path, "r")
    n_traj = len(traj)
    n_df   = len(df_full)

    if n_traj == 0:
        raise RuntimeError(f"Empty trajectory: {traj_path}")
    if n_df < n_traj:
        raise RuntimeError(f"CSV shorter than trajectory? n_df={n_df}, n_traj={n_traj}")

    # infer stride (e.g. 10) from lengths
    stride = max(1, int(round(n_df / n_traj)))

    # downsample df to match traj sampling
    df_traj = df_full.iloc[::stride].copy()

    # trim/pad to exactly n_traj (trim is typical)
    if len(df_traj) > n_traj:
        df_traj = df_traj.iloc[:n_traj]
    elif len(df_traj) < n_traj:
        # rare: if rounding undershot, take the last n_traj rows evenly spaced
        idx = np.linspace(0, n_df - 1, n_traj).round().astype(int)
        df_traj = df_full.iloc[idx].copy()

    # sanity
    if len(df_traj) != n_traj:
        raise RuntimeError(f"Alignment failed: len(df_traj)={len(df_traj)} vs len(traj)={n_traj}")

    return traj, df_traj, stride

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
    N=576
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
    LLPT_Tdict={1700:np.arange(186,201,2), 1800: np.arange(180,191),2000: np.arange(160,181), 
                2100: np.arange(156,171,2), 2300: np.arange(140,161,2),2500:np.arange(130,151,2)}
    #LLPT_Tdict2={1700:np.arange(188,196.1,0.5), 1800:np.arange(182,185.1,0.2), 2000: np.arange(165.5,171.6,0.5),
    #             2100:np.arange(158,164.1,0.5), 2300:np.arange(143,152.1,0.5), 2500:np.arange(136.5,141.1)}
    LLPT_Tdict2={1700:np.arange(188,197,1), 2100:np.arange(158,165,1), 2300:np.arange(143,153,1), 2500:np.arange(136,142)}

    N = 576
    Ang2bohr = 1.88973
    trange = LLPT_Tdict.keys()
    all_p = np.concatenate([LLPT_Tdict[t] for t in trange])
    molperc_all = get_all_atomicity_data_ASE(all_p)
    norm = Normalize(vmin=min(molperc_all),vmax=max(molperc_all))
    sm = ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])
    #trange=[2000,2100,2300,2500]
    trange=[2300,2500]
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
        p_to_add = np.unique(np.concatenate((p1_to_add,p2_to_add)))
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
            analysis_traj_path = f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{pgpa}/liquid/NPT_ASE/{tkel}/analysis/P{pgpa}T{tkel}.merged.traj"
            traj, df_traj, stride = align_df_to_traj(df, analysis_traj_path, time_col="time")
            mask = df_traj["time"].to_numpy() > 1.0
            if mask.sum() < 500:
                print(f"Not enough (traj-aligned) data for P={pgpa} GPa,T={tkel} K")
                continue
            
            #traj = io.read(analysis_traj_path,index="10:")
            #vol = np.array([atoms.get_volume() for atoms in traj])
            #traj = Trajectory(analysis_traj_path,'r')
            idx = np.nonzero(mask)[0]
            vol = np.fromiter((traj[i].get_volume() for i in idx), dtype=float, count=len(idx))
            #vol = np.fromiter((traj[i].get_volume() for i in range(len(traj))),dtype=float,count=len(traj))
            # df = df[df['time']>1]
            # if len(df)<500:
            #     print(f"Not enough data for P={pgpa} GPa,T={tkel} K")
            #     continue
            df2 = df_traj.loc[mask]
            Press_val = -1.0*(df2["Pxx"]+df2["Pyy"]+df2["Pzz"])/3
            press_blocked = reblock(Press_val.values,100)
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
            #molperc = get_atomic_info_from_PT_ASE(pgpa,tkel,'liquid')
            #mperc.append(molperc)
            mperc.append(float(get_atomic_info_from_PT_ASE(pgpa, tkel, "liquid")))
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
    N=576
    # prange = [int(os.path.basename(name)[1:]) for name in glob.glob(f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p*")]
    # tv_eos(prange)
    #pv_eos(case="NPT_mliap")
    delta_F2_all, delta_V_all = pf_eos(case="NPT_mliap")  # Pressure vs c_total_force_sq/N
    #delta_F2_all, delta_V_all = pf_eos_split_by_means(case="NPT_mliap")  # Pressure vs c_total_force_sq/N
    # print(delta_F2_all)
    # print(delta_V_all)
    # #Calculate dP/dx
    mass = 1.008 #dalton
    hbar = 1973 #eV A
    mass = mass*931494697.25613 #dalton to eV
    kb=8.61733*10**(-5)
    lamb_ = hbar**2/(2*mass) #eV A^2
    dpdx_all = {}
    for Tm in delta_F2_all.keys():
        Ang_to_bohr = 1.88973
        slope_classical = 1.0/12*delta_F2_all[Tm]/(kb*Tm)**2
        dpdx = -slope_classical*lamb_/delta_V_all[Tm]
        dpdx_all[Tm] = dpdx*160.21766208 #eV/A^3 to GPa
        print(f"dP/dx for T={Tm} K is {dpdx_all[Tm]} GPa")
    np.savetxt(f'{N}atoms_dpdx_all_whistogramfit.txt',np.c_[list(dpdx_all.keys()),list(dpdx_all.values())],header='Temperature(K)  dP/dx')
    return dpdx_all

if __name__ == '__main__':
    main()
