import numpy as np
import matplotlib.pyplot as plt


# Load data -- from /work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p150/liquid/NPT_mliap/2100/analysis/P150T2100gofr_info_LAMMPS_liquid.npy
#and /work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p150/liquid/NPT/2100/analysis/P150T2100gofr_info_LAMMPS_liquid.npy and plot them on top of each other

def gofr_comparison(press,temp):
    data_mliap = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/NPT_mliap/{temp}/analysis/P{press}T{temp}gofr_info_LAMMPS_liquid.npy')
    data_m29_lammps = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/NPT/{temp}/analysis/P{press}T{temp}gofr_info_LAMMPS_liquid.npy')
    #make side by side plots
    r = data_mliap[:,0]
    g_r_mliap = data_mliap[:,1]
    g_r_err_mliap = data_mliap[:,2]
    g_r_m29_lammps = data_m29_lammps[:,1]
    g_r_m29_err = data_m29_lammps[:,2]

    #plot difference of gofr wrt r 
    g_r_diff = g_r_mliap - g_r_m29_lammps
    g_r_diff_err = np.sqrt(g_r_err_mliap**2 + g_r_m29_err**2)


    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    # --- Left: overlay g(r) from both models
    axes[0].errorbar(r, g_r_mliap, yerr=g_r_err_mliap,
                    label='ML-IAP', color='blue')
    axes[0].errorbar(r, g_r_m29_lammps, yerr=g_r_m29_err,
                    label='M29 LAMMPS', color='orange', linestyle='dashed')
    axes[0].set_xlabel('r (Å)')
    axes[0].set_ylabel('g(r)')
    axes[0].set_title(f'g(r) Comparison\nP={press} GPa, T={temp} K')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Right: difference plot
    axes[1].plot(r, g_r_diff, label='ML-IAP - M29 LAMMPS', color='green')
    axes[1].fill_between(r,
                        g_r_diff - g_r_diff_err,
                        g_r_diff + g_r_diff_err,
                        color='green', alpha=0.25,
                        label='1σ uncertainty')
    axes[1].set_xlabel('r (Å)')
    axes[1].set_ylabel('Δg(r)')
    axes[1].set_title('Difference in g(r)')
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'compare_and_difference_gofr_P{press}_T{temp}.png')
    plt.close()

def gofr_leonardo_comparison(press,temp):
    data_leonardo_mliap = np.loadtxt(f"/work/hdd/bcqo/isaitov/HYDROGEN/LEONARDO/N360/T{temp}/NPT_{press}GPa/rdf_final.txt")
    data_m29_lammps = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/NPT/{temp}/analysis/P{press}T{temp}gofr_info_LAMMPS_liquid.npy')
    r_leonardo = data_leonardo_mliap[:,1]
    g_r_leonardo = data_leonardo_mliap[:,2]
    r_m29 = data_m29_lammps[:,0]
    g_r_m29_lammps = data_m29_lammps[:,1]
    g_r_m29_err = data_m29_lammps[:,2]
    #align the two histograms of r_leonardo and g_r_leonardo and g_r_m29_lammps data and then subtract to obtain the difference.each element is a bin edge so we should be able to collect within a unified bin
    g_r_m29_lammps_interp = np.interp(r_leonardo, r_m29, g_r_m29_lammps)
    g_r_diff_leonardo = g_r_leonardo - g_r_m29_lammps_interp
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    # --- Left: overlay g(r) from both models
    axes[0].plot(r_leonardo, g_r_leonardo,
                    label='ML-IAP', color='blue')
    axes[0].errorbar(r_m29, g_r_m29_lammps, yerr=g_r_m29_err,
                    label='M29 LAMMPS', color='orange', linestyle='dashed')
    axes[0].set_xlabel('r (Å)')
    axes[0].set_ylabel('g(r)')
    axes[0].set_title(f'g(r) Comparison\nP={press} GPa, T={temp} K')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Right: difference plot
    axes[1].plot(r_leonardo, np.abs(g_r_diff_leonardo), label='|ML-IAP - M29 LAMMPS|', color='green')
    axes[1].set_xlabel('r (Å)')
    axes[1].set_ylabel('|Δg(r)|')
    axes[1].set_title('Difference in g(r)')
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'compare_and_difference_leonardo_ilnur_gofr_P{press}_T{temp}.png')
    plt.close()

def gofr_leonardo_delta_comparison(press,temp):
    data_leonardo_mliap = np.loadtxt(f"/work/hdd/bcqo/isaitov/HYDROGEN/LEONARDO/N360/T{temp}/NPT_{press}GPa/rdf_final.txt")
    if press==150:
        case='NPT_mliap_2'
    else:
        case='NPT_mliap'   
    data_delta_mliap = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/{case}/{temp}/analysis/P{press}T{temp}gofr_info_LAMMPS_liquid.npy')
    r_leonardo = data_leonardo_mliap[:,1]
    g_r_leonardo = data_leonardo_mliap[:,2]
    r_delta = data_delta_mliap[:,0]
    g_r_delta = data_delta_mliap[:,1]
    g_r_delta_err = data_delta_mliap[:,2]
    #align the two histograms of r_leonardo and g_r_leonardo and g_r_m29_lammps data and then subtract to obtain the difference.each element is a bin edge so we should be able to collect within a unified bin
    g_r_delta_interp = np.interp(r_leonardo, r_delta, g_r_delta)
    g_r_diff_leonardo = g_r_leonardo - g_r_delta_interp
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    # --- Left: overlay g(r) from both models
    axes[0].plot(r_leonardo, g_r_leonardo,
                    label='Leonardo', color='blue')
    axes[0].errorbar(r_delta, g_r_delta, yerr=g_r_delta_err,
                    label='Delta', color='orange', linestyle='dashed')
    axes[0].set_xlabel('r (Å)')
    axes[0].set_ylabel('g(r)')
    axes[0].set_title(f'g(r) Comparison\nP={press} GPa, T={temp} K')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Right: difference plot
    axes[1].plot(r_leonardo, np.abs(g_r_diff_leonardo), label='|Leonardo - Delta|', color='green')
    axes[1].set_xlabel('r (Å)')
    axes[1].set_ylabel('|Δg(r)|')
    axes[1].set_title('Difference in g(r)')
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'compare_and_difference_leonardo_delta_gofr_P{press}_T{temp}.png')
    plt.close()

def gofr_leonardo_ASEdelta_comparison(press,temp):
    data_leonardo_mliap = np.loadtxt(f"/work/hdd/bcqo/isaitov/HYDROGEN/LEONARDO/N360/T{temp}/NPT_{press}GPa/rdf_final.txt")
    data_m29_ase = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/NPT_ASE/{temp}/analysis/P{press}T{temp}gofr_info_ASE_liquid.npy')
    r_leonardo = data_leonardo_mliap[:,1]
    g_r_leonardo = data_leonardo_mliap[:,2]
    r_m29 = data_m29_ase[:,0]
    g_r_m29_ase = data_m29_ase[:,1]
    g_r_m29_err = data_m29_ase[:,2]
    #if len(r_leonardo) != len(r_m29):
    #align the two histograms of r_leonardo and g_r_leonardo and g_r_m29_lammps data and then subtract to obtain the difference.each element is a bin edge so we should be able to collect within a unified bin
    g_r_m29_ase_interp = np.interp(r_leonardo, r_m29, g_r_m29_ase)
    #g_r_m29_err_interp = np.interp(r_leonardo,r_m29,g_r_m29_err)
    g_r_diff_leonardo = g_r_leonardo - g_r_m29_ase_interp
    # else:
    #     g_r_diff_leonardo = g_r_leonardo - g_r_m29_ase
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    # --- Left: overlay g(r) from both models
    axes[0].plot(r_leonardo, g_r_leonardo,
                    label='ML-IAP', color='blue')
    axes[0].errorbar(r_m29, g_r_m29_ase, yerr=g_r_m29_err,
                    label='M29 ASE', color='orange', linestyle='dashed')
    axes[0].set_xlabel('r (Å)')
    axes[0].set_ylabel('g(r)')
    axes[0].set_title(f'g(r) Comparison\nP={press} GPa, T={temp} K')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Right: difference plot
    axes[1].plot(r_leonardo, np.abs(g_r_diff_leonardo), label='|ML-IAP - M29 LAMMPS|', color='green')
    axes[1].fill_between(r_leonardo,
                        np.abs(g_r_diff_leonardo) - np.sqrt(np.sum(g_r_m29_err**2)/len(g_r_m29_err)),
                        np.abs(g_r_diff_leonardo) + np.sqrt(np.sum(g_r_m29_err**2)/len(g_r_m29_err)),
                        color='green', alpha=0.25,
                        label='M29 Root mean square uncertainty only')
    axes[1].set_xlabel('r (Å)')
    axes[1].set_ylabel('|Δg(r)|')
    axes[1].set_title('Difference in g(r)')
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'compare_and_difference_with_ASE_leonardo_ilnur_gofr_P{press}_T{temp}.png')
    plt.close()

def gofr_leonardo_ASEdelta_comparison_wupdatedmliap(press,temp):
    data_leonardo_mliap = np.loadtxt(f"/work/hdd/bcqo/isaitov/HYDROGEN/LEONARDO/N360/T{temp}/NPT_{press}GPa/rdf_final.txt")
    data_m29_ase = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/NPT_ASE/{temp}/analysis/P{press}T{temp}gofr_info_ASE_liquid.npy')
    data_m29_mliap = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/NPT_mliap_new/{temp}/analysis/P{press}T{temp}gofr_info_LAMMPS_liquid.npy')
    r_leonardo = data_leonardo_mliap[:,1]
    g_r_leonardo = data_leonardo_mliap[:,2]
    r_m29_ase = data_m29_ase[:,0]
    r_m29_mliap = data_m29_mliap[:,0]
    g_r_m29_ase = data_m29_ase[:,1]
    g_r_m29_err_ase = data_m29_ase[:,2]
    g_r_m29_mliap = data_m29_mliap[:,1]
    g_r_m29_err_mliap = data_m29_mliap[:,2]
    #if len(r_leonardo) != len(r_m29):
    #align the two histograms of r_leonardo and g_r_leonardo and g_r_m29_lammps data and then subtract to obtain the difference.each element is a bin edge so we should be able to collect within a unified bin
    g_r_m29_ase_interp = np.interp(r_leonardo, r_m29_ase, g_r_m29_ase)
    g_r_m29_mliap_interp = np.interp(r_leonardo, r_m29_mliap, g_r_m29_mliap)
    g_r_diff_leonardo = g_r_leonardo - g_r_m29_ase_interp
    g_r_diff_leonardo_mliap = g_r_leonardo - g_r_m29_mliap_interp
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    # --- Left: overlay g(r) from both models
    axes[0].plot(r_leonardo, g_r_leonardo,
                    label='ML-IAP', color='blue')
    axes[0].errorbar(r_m29_ase, g_r_m29_ase, yerr=g_r_m29_err_ase,
                    label='M29 ASE', color='orange', linestyle='dashed')
    axes[0].errorbar(r_m29_mliap, g_r_m29_mliap, yerr=g_r_m29_err_mliap,
                    label='M29 LAMMPS', color='purple', linestyle='dashed')
    axes[0].set_xlabel('r (Å)')
    axes[0].set_ylabel('g(r)')
    axes[0].set_title(f'g(r) Comparison\nP={press} GPa, T={temp} K')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Right: difference plot
    axes[1].plot(r_leonardo, np.abs(g_r_diff_leonardo), label='|ML-IAP - M29 ASE|', color='green')
    axes[1].plot(r_leonardo, np.abs(g_r_diff_leonardo_mliap), label='|ML-IAP - M29 LAMMPS|', color='red')
    axes[1].fill_between(r_leonardo,
                        np.abs(g_r_diff_leonardo) - np.sqrt(np.sum(g_r_m29_err_ase**2)/len(g_r_m29_err_ase)),
                        np.abs(g_r_diff_leonardo) + np.sqrt(np.sum(g_r_m29_err_ase**2)/len(g_r_m29_err_ase)),
                        color='green', alpha=0.25,
                        label='M29 ASE Root mean square uncertainty only')
    axes[1].fill_between(r_leonardo,
                        np.abs(g_r_diff_leonardo_mliap) - np.sqrt(np.sum(g_r_m29_err_mliap**2)/len(g_r_m29_err_mliap)),
                        np.abs(g_r_diff_leonardo_mliap) + np.sqrt(np.sum(g_r_m29_err_mliap**2)/len(g_r_m29_err_mliap)),
                        color='red', alpha=0.25,
                        label='M29 LAMMPS Root mean square uncertainty only')
    axes[1].set_xlabel('r (Å)')
    axes[1].set_ylabel('|Δg(r)|')
    axes[1].set_title('Difference in g(r)')
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'compare_and_difference_with_ASE_and_LAMMPS_leonardo_ilnur_gofr_P{press}_T{temp}.png')
    plt.close()


def sofk_comparison(press,temp):
    data_mliap = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/NPT_mliap/{temp}/analysis/P{press}T{temp}liquidsofk.txt')
    data_m29_lammps = np.loadtxt(f'/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms/p{press}/liquid/NPT/{temp}/analysis/P{press}T{temp}liquidsofk.txt')

    k = data_mliap[:,0]
    S_k_mliap = data_mliap[:,1]
    S_k_err_mliap = data_mliap[:,2]
    S_k_m29_lammps = data_m29_lammps[:,1]
    S_k_m29_err = data_m29_lammps[:,2]
    plt.errorbar(k, S_k_mliap, yerr=S_k_err_mliap, label='ML-IAP', color='blue')
    plt.errorbar(data_m29_lammps[:,0], S_k_m29_lammps, yerr=S_k_m29_err, label='M29 LAMMPS', color='orange', linestyle='dashed')
    plt.xlabel('k (1/Angstrom)')
    plt.ylabel('S(k)')
    plt.title(f'Comparison of S(k) from ML-IAP and M29 LAMMPS at P={press} GPa, T={temp} K')
    plt.legend()
    plt.savefig(f'compare_sofk_mliap_m29_lammps_P{press}_T{temp}.png')
    plt.close()


#gofr_leonardo_comparison(150,2100)
#sofk_comparison(150,2100)
# gofr_comparison(162,2100)
# sofk_comparison(162,2100)
#gofr_comparison(170,2100)
gofr_leonardo_ASEdelta_comparison_wupdatedmliap(150,2100)
#gofr_leonardo_ASEdelta_comparison(170,2100)
#sofk_comparison(170,2100)
#gofr_leonardo_delta_comparison(150,2100)
#gofr_leonardo_delta_comparison(170,2100)
