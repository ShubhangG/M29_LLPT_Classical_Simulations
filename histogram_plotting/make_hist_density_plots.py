import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys,os,glob
from scipy.optimize import minimize
from scipy.special import logsumexp
import pickle as pkl

def define_energy_volume_grid(P,T,N,bins=100):
    data_path=f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{P}/liquid/NPT/{T}/analysis"
    data = pd.read_csv(f"{data_path}/P{P}T{T}M29_merged_lammps_out.csv")
    data = data[data['Step']>1000]
    V = data['Volume'].values
    E = data['PotEng'].values 
    total_hist,yedges,xedges = np.histogram2d(V,E,bins=bins,density=False)
    plt.pcolormesh(xedges,yedges,total_hist,cmap='Reds')
    plt.colorbar()
    plt.xlabel('Volume ($A^3$)')
    plt.ylabel('Energy (eV)')
    plt.title(f"P={P} GPa, T={T} K")
    plt.savefig(f"{data_path}/P{P}T{T}_hist_VE.png",dpi=200)
    plt.close() 

    return total_hist,xedges,yedges,V,E
    
def ev_histgrid(pressure_range=None,temperature_range=None):
    hist_per_simu = {}
    N_list = [576,1200]
    # if os.path.exists("hist_plots/hist_NPT_dict.pkl"):
    #     with open("hist_plots/hist_NPT_dict.pkl","rb") as f:
    #         hist_per_simu = pkl.load(f)
    volumes_edges = {}
    energies_edges = {}
    all_hist = {}

    for N in N_list:
        volumes_all = [] 
        energies_all = []
        p_path= f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/"
        #print([os.path.basename(p)[1:] for p in glob.glob(f"{p_path}/p*")])
        #exit()
        if pressure_range is not None:
            all_p = pressure_range
        else:
            all_p = [float(os.path.basename(p)[1:]) for p in glob.glob(f"{p_path}/p*")]
            #pressure_range = [min(all_p),max(all_p)]
        for p in all_p:
            if p.is_integer():
                p=int(p) 
            if temperature_range is not None:
                all_t = temperature_range
            else:
                all_t = np.array(os.listdir(f"{p_path}/p{p}/liquid/NPT/"),dtype=int)
            for t in all_t:
                try:
                    tothist,Vedges,Eedges,V,E = define_energy_volume_grid(p,t,N)
                except Exception as e:
                    print(f"Could not get data for P={p} GPa,T={t} K")
                    continue
                volumes_all.extend(V)
                energies_all.extend(E)
                hist_per_simu[(N,p,t)] = tothist
            

        total_all_hist, yedges_all,xedges_all = np.histogram2d(volumes_all,energies_all,bins=100,density=False)
        plt.pcolormesh(xedges_all,yedges_all,total_all_hist,cmap='Reds')
        plt.xlabel('Volume ($A^3$)')
        plt.ylabel('Energy (eV)')
        plt.title(f"All P,T combined for N={N}")
        plt.colorbar()
        if pressure_range is None:
            plt.savefig(f"hist_plots/all_{N}atoms_P_T_hist_VE_overall.png",dpi=200)
        else:
            plt.savefig(f"hist_plots/all_{N}atoms_P_T_hist_VE_prange{pressure_range[0]}_{pressure_range[1]}_trange{temperature_range[0]}_{temperature_range[1]}.png",dpi=200)
        plt.close()
        volumes_edges[N] = xedges_all
        energies_edges[N] = yedges_all
        all_hist[N] = total_all_hist

    # with open("hist_plots/hist_NPT_dict.pkl","wb") as f:
    #     pkl.dump(hist_per_simu,f)

    # with open("hist_plots/hist_NPT_allhist.pkl","wb") as f:
    #     pkl.dump((all_hist,volumes_edges,energies_edges),f)

    return volumes_edges,energies_edges,all_hist,hist_per_simu

def find_zeros_hist_E_V(hist_per_simu, threshold=1):
    """
    given histograms of each simulation,
    returns a mask with True if there is a non zero value of the histogram at the given (E, V)
    """
    total_hist = 0
    for hist_NPT in hist_per_simu.values():
        total_hist += hist_NPT # adding an array so its not a number after the first iteration
    return total_hist < threshold

def histogram_direct_iteration(N,n_iter=100,pressure_range=None,temperature_range=None):
    kb = 8.617333 * 1e-5
    evA3_to_GPa = 160.2
    # if os.path.exists("hist_plots/hist_NPT_dict.pkl"):
    #     with open("hist_plots/hist_NPT_dict.pkl","rb") as f:
    #         hist_per_simu = pkl.load(f)
    
    # if os.path.exists("hist_plots/hist_NPT_allhist.pkl"):
    #     with open("hist_plots/hist_NPT_allhist.pkl","rb") as f:
    #         all_hist,volume_edges,energies_edges = pkl.load(f)
    # else:
    volume_edges,energies_edges,all_hist,hist_per_simu = ev_histgrid(pressure_range=pressure_range,temperature_range=temperature_range)
    
    Vedges = volume_edges[N]
    Eedges = energies_edges[N]
    all_hist_N = all_hist[N]
    hist_per_simu_N = {k:v for k,v in hist_per_simu.items() if k[0]==N}
    Vcenters = 0.5*(Vedges[1:]+Vedges[:-1])
    Ecenters = 0.5*(Eedges[1:]+Eedges[:-1])
    volume_mesh, energy_mesh = np.meshgrid(Vcenters, Ecenters, indexing='ij')
    log_omega = np.zeros_like(all_hist_N, dtype=np.longdouble)
    #remove zero entries to avoid log(0)
    mask_zeros = find_zeros_hist_E_V(hist_per_simu, threshold=10)
    log_omega[mask_zeros] = -np.inf
    log_omega_next = log_omega.copy()
    total_hist = sum([h for h in hist_per_simu_N.values()])

    diffs = []
    mask_non_zeros = np.logical_not(mask_zeros)
    if pressure_range == None:
        pressure_range = sorted(list(set([k[1] for k in hist_per_simu_N.keys()])))
    if temperature_range == None:
        temperature_range = sorted(list(set([k[2] for k in hist_per_simu_N.keys()])))
    for i in range(n_iter):
        beta_e_pv_Zs = []
        for (N_i, press_i_GPa, temp_i) in hist_per_simu_N.keys():   
            # inverse temperature
            beta_i = 1 / (kb * temp_i)
            press_i_ev = press_i_GPa / evA3_to_GPa

            # compute
            beta_e_pv_i =  beta_i * (energy_mesh + press_i_ev * volume_mesh)
            log_Z_i = logsumexp(log_omega[mask_non_zeros] - beta_e_pv_i[mask_non_zeros])

            # store terms present in exponential in big list
            sample_points = np.sum(hist_per_simu_N[(N_i, press_i_GPa, temp_i)])
            beta_e_pv_Zs.append(np.log(sample_points) - beta_e_pv_i - log_Z_i)

        log_denominator = np.zeros_like(log_omega)
        for i_e in range(log_omega.shape[0]):
            for j_v in range(log_omega.shape[1]):
                if mask_zeros[i_e, j_v]:
                    continue
                # instead of having 
                e_pv_i_j = [beta_e_pv_Z[i_e, j_v] for beta_e_pv_Z in beta_e_pv_Zs]
                log_denominator[i_e, j_v] = logsumexp(e_pv_i_j)

        log_omega_next = np.log(total_hist[mask_non_zeros]) - log_denominator[mask_non_zeros]
        diffs.append(np.sum((log_omega_next - log_omega[mask_non_zeros])**2)) 
        log_omega[mask_non_zeros] = log_omega_next.copy()
        print(f"Iteration {i}: {diffs[-1]}")

    # save both grid and edges eventhough it contains redundant information.
    density = {
        'log_omega': log_omega,
        'energy_mesh': energy_mesh,
        'energy_edges': Eedges,
        'volume_mesh': volume_mesh,
        'volume_edges': Vedges,
        'mask_zeros': mask_zeros,
        'pressure_range': pressure_range,
        'temperature_range': temperature_range,
        'hist_per_simu': hist_per_simu,
    }

    return diffs, density


def prob_fitted(density, E, vol, press_GPa, temp, kb=8.617333 * 1e-5, evA3_to_GPa=160.2, log_Z=None):
    """
    given the probability funciton fitted with multiple histograms,
    compute the probability of (E, V, beta, P).
    density is a dictionnary with keys
    'log_omega', 'energy_mesh', 'volume_mesh' and 'mask_zeros'.
    Note that the probability is not computed exactly for the (E, V) given,
    but for the closest value on the grid (otherwise the probability might be
    bigger than 1.
    """
    beta = 1 / (kb * temp)
    press_ev = press_GPa / evA3_to_GPa

    mask_zeros = density['mask_zeros']

    e_mesh, vol_mesh = density['energy_mesh'], density['volume_mesh']
    # since e_mesh is a mesh, a full column is enough to have all values
    energy_vals = e_mesh[0, :]
    volume_vals = vol_mesh[:, 0]
    # test if energy is outside bounds:
    energy_outside = E < np.min(energy_vals) or E > np.max(energy_vals)
    volume_outside = vol < np.min(volume_vals) or E > np.max(volume_vals)
    if energy_outside or volume_outside:
        grid = f'{np.min(energy_vals)}, {np.max(energy_vals)}, {np.min(volume_vals)}, {np.max(volume_vals)}'
        print(f'energy {E} or {vol} outside of (E, V) grid: ( {grid} )' )

    e_idx = np.argmin(np.abs(energy_vals - E))
    vol_idx = np.argmin(np.abs(volume_vals - vol))

    # energy in calculation should not be the (E, V) given, but rounded to energy and volume defined by the grid
    # otherwise probability can be bigger than 1!
    E_val, vol_val = energy_vals[e_idx], volume_vals[vol_idx]
    beta_e_pv = beta * (E_val + press_ev * vol_val)

    # define log_Z on the same grid as enregy and volume
    if log_Z is None:
        beta_e_pv_grid =  beta * (density['energy_mesh'] + press_ev * density['volume_mesh'])
        log_Z = logsumexp(density['log_omega'][np.logical_not(mask_zeros)] - beta_e_pv_grid[np.logical_not(mask_zeros)])

    log_prob = density['log_omega'][vol_idx, e_idx] - beta_e_pv - log_Z
    prob = np.exp(log_prob, dtype=np.longdouble)
    return prob, log_Z

def evaluate_prob(density, press_GPa, temp):
    """
    Evaluate probability distribution Eq. S2 P(E, V, beta, P)
    on all points of the grid (E, V)

    """
    # check that pressure and temperature are not outside bounds, raise warnings otherwise
    cond_press = press_GPa > density['pressure_range'][1] or press_GPa < density['pressure_range'][0]
    cond_temp = temp > density['temperature_range'][1] or temp < density['temperature_range'][0]
    if cond_press or cond_temp:
        print(f"pressure: {press_GPa} or/and temp: {temp} is outside the fitting range {density['pressure_range']} and {density['temperature_range']}")

    e_mesh, vol_mesh = density['energy_mesh'], density['volume_mesh']
    # since e_mesh is a mesh, a full column is enough to have all values
    energy_vals = e_mesh[0, :]
    volume_vals = vol_mesh[:, 0]
    
    # Initialize a 2D array to store the results
    prob_grid = np.zeros((len(energy_vals), len(volume_vals)))

    log_Z = None
    for i, E in enumerate(energy_vals):
        for j, vol in enumerate(volume_vals):
            prob_grid[i, j], log_Z = prob_fitted(density, E, vol, press_GPa, temp, log_Z=log_Z)

    return energy_vals, volume_vals, prob_grid


def volume_to_rs(volume, N):
    """
    Convert volume in Angstrom^3 to Wigner-Seitz radius in bohr
    for a system of N atoms.
    """
    Ang2bohr = 1.88973
    volume_per_atom = volume / N
    rs = (3 / (4 * np.pi) * volume_per_atom)**(1/3) * Ang2bohr
    return rs

def mean_energy_volume(density, N_select, press_GPa, temp, qty='energy'):
    energy_vals, volume_vals, prob_grid = evaluate_prob(density, press_GPa, temp)
    key_mesh = 'energy_mesh' if qty == 'energy' else 'volume_mesh'
    if qty == 'energy':
        return np.sum(density[key_mesh] / N_select * prob_grid.T)
    else:
        return np.sum(density[key_mesh] * prob_grid.T)

def equation_of_state_from_histogram(density,N,tkel,prange):
    mean_rs_hist= []
    prange_ = np.linspace(np.min(prange),np.max(prange),100)
    for p in prange_:
        mean_vol = mean_energy_volume(density,N,p,tkel,qty='volume')
        mean_rs_hist.append(volume_to_rs(mean_vol,N))

    return prange_,mean_rs_hist 

def cv_from_histogram(density,N,tkel,prange):
    cv_hist = []
    prange_ = np.linspace(np.min(prange),np.max(prange),100)
    for p in prange_:
        var_e = compute_energy_volume_variance(density,p,tkel,qty='energy')
        cv = var_e/N
        cv_hist.append(cv)

    return prange_,cv_hist

def compute_energy_volume_variance(density, press_GPa, temp, qty='energy'):
    energy_vals, volume_vals, prob_grid = evaluate_prob(density, press_GPa, temp)

    key_mesh = 'energy_mesh' if qty == 'energy' else 'volume_mesh'

    # create grid to vectorize mean and variance computations
    grid = density[key_mesh]

    # careful, the transpose of the probability grid
    mean_qty = np.sum(grid * prob_grid.T)
    var_qty = np.sum(grid**2 * prob_grid.T) - mean_qty**2
    return var_qty



def plot_eos_w_hist_reweighting(density,N,trange=None,pressure_range=None):
    pflag = False
    if trange is None:
        trange = np.arange(1700,2501,100)
    if pressure_range is None:
        pflag = True        
    for tkel in trange:
        try:
            eos_data_from_sim = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/EoS_plots/eos_liquid_{N}atoms_{tkel}K.txt')
            rs_sim = eos_data_from_sim[:,0]
            p_sim = eos_data_from_sim[:,1]
            l=plt.errorbar(rs_sim,p_sim,xerr=eos_data_from_sim[:,2],yerr=eos_data_from_sim[:,3],fmt='o')
            if pflag:
                pressure_range = p_sim
            phist, mean_rs_hist = equation_of_state_from_histogram(density,N,tkel,pressure_range)
            plt.plot(mean_rs_hist,phist,'-',color=l.color)    
        except Exception as e:
            print(f"Could not find data for T={tkel} K")
            if pflag:
                pressure_range = np.linspace(140,200,100)
            phist, mean_rs_hist = equation_of_state_from_histogram(density,N,tkel,pressure_range)
            plt.plot(mean_rs_hist,phist,'-')
        
    plt.xlabel('rs (bohr)')
    plt.ylabel('Pressure (GPa)')
    plt.title(f'Equation of State from histogram reweighting N={N}')
    plt.legend()
    plt.savefig(f'hist_plots/eos_histogram_reweighting_{N}atoms_{trange[0]}to{trange[-1]}.png',dpi=200)
    plt.close()

def plot_variance_energy_volume(density,N,trange=None,pressure_range=None):
    pflag = False
    if trange is None:
        trange = np.arange(1700,2501,100)
    if pressure_range is None:
        pflag = True
    for tkel in trange:
        try:
            cv_data_from_sim = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/Cv_plots/Cv_liquid_{N}atoms_{tkel}K_data.txt')
            p_sim = cv_data_from_sim[:,0]
            cv_sim = cv_data_from_sim[:,1]
            l=plt.errorbar(p_sim,cv_sim,fmt='o')
            if pflag:
                pressure_range = p_sim
            phist, cv_hist = cv_from_histogram(density,N,tkel,pressure_range)
            plt.plot(phist,cv_hist,'-',color=l.color)
        except Exception as e:
            print(f"Could not find data for T={tkel} K")
            if pflag:
                pressure_range = np.linspace(140,200,100)
            phist, cv_hist = cv_from_histogram(density,N,tkel,pressure_range)
            plt.plot(phist,cv_hist,'-')


    plt.ylabel('Cv')
    plt.xlabel('Pressure (GPa)')
    plt.title(f'Cv from histogram reweighting N={N}')
    plt.legend()
    plt.savefig(f'hist_plots/cv_histogram_reweighting_{N}atoms_{trange[0]}to{trange[-1]}.png',dpi=200)
    plt.close()

def comparison_simulation_histogram(density, hist_per_simu, N_select, press_GPa, temp):
    Ang_to_bohr = 1 / 0.5291

    energy_vals, volume_vals, prob_grid = evaluate_prob(density, press_GPa, temp)
    energy_edges = density['energy_edges']
    volume_edges = density['volume_edges']

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.tight_layout()
    E_grid, vol_grid = np.meshgrid(energy_vals, volume_vals, indexing='ij')
    rs_grid = np.power(3 / (4 * np.pi) * vol_grid * Ang_to_bohr**3 / N_select, 1.0/3)
    # pc = axs[0].pcolormesh(E_grid / N_select, rs_grid, prob_grid, shading='auto', cmap='Reds')
    pc = axs[0].pcolormesh(E_grid / N_select, vol_grid, prob_grid, shading='auto', cmap='Reds')
    fig.colorbar(pc, label='Probability')
    axs[0].set_ylabel('Volume')
    axs[0].set_title(f'reconstruction p={press_GPa}, T={temp}')
    axs[0].grid()

    axs[1].set_title(f'simulation p={press_GPa}, T={temp}')
    prob = hist_per_simu[(N_select, press_GPa, temp)] / np.sum(hist_per_simu[(N_select, press_GPa, temp)])
    rs_edges = np.power(3 / (4 * np.pi) * volume_edges * Ang_to_bohr**3 / N_select, 1 / 3)
    # pc2 = axs[1].pcolormesh(energy_edges / N_select, rs_edges, prob, cmap='Reds')
    pc2 = axs[1].pcolormesh(energy_edges / N_select, volume_edges, prob, cmap='Reds')
    fig.colorbar(pc2, label='Probability')
    axs[1].grid()
    axs[1].set_xlabel('Energy')
    plt.savefig(f'hist_plots/comparison_simulation_histogram_N{N_select}_P{press_GPa}_T{temp}.png', dpi=200)
    plt.close()

def main():
    volume_edges,energies_edges,all_hist,hist_per_simu=ev_histgrid()
    with open("hist_data/ev_histgrid_all_NPT.pkl","wb") as f:
        pkl.dump((volume_edges,energies_edges,all_hist,hist_per_simu),f)
    # N_atoms_list = [576]
    # densities = {}
    # pressure_range = [171,170,169,168]
    # temperature_range = [2000,2100]
    # for N in N_atoms_list:
    #     if os.path.exists(f"hist_data/histogram_densities.pkl"):
    #         with open(f"hist_data/histogram_densities.pkl","rb") as f:
    #             densities = pkl.load(f)
    #         density=densities[N]
    #     else:       
    #         diffs, density = histogram_direct_iteration(N,n_iter=500,pressure_range=pressure_range,temperature_range=temperature_range)
    #     #plt.semilogy(diffs[:], label=f'N atoms={N}')
    #         densities[N] = density

    #     comparison_simulation_histogram(density, density['hist_per_simu'], N, 169, 2000)

    #     #energy_vals, volume_vals, prob_grid = evaluate_prob(density, 175, 2000)

    #     plot_eos_w_hist_reweighting(density,N,temperature_range,pressure_range)
    #     plot_variance_energy_volume(density,N,temperature_range,pressure_range)



    # with open("hist_data/histogram_densities.pkl","wb") as f:
    #     pkl.dump(densities,f)

    return
if __name__ == "__main__":
    main() 