import numpy as np
import matplotlib.pyplot as plt
import os


def PV_eos_plot(atom_sizes,temps):
    for temp in temps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # EoS plot
        for i,N in enumerate(atom_sizes):
            if N>1200:
                M29_eos_path = f"./Ilnur_data/eos_liquid_{N}atoms_{temp}K.txt"
                if os.path.exists(M29_eos_path):
                    EOS_data = np.loadtxt(M29_eos_path)
                    axes[0].errorbar(EOS_data[:,0], EOS_data[:,2], xerr=EOS_data[:,1], yerr=EOS_data[:,3], fmt='--', label=f"M29 {N} atoms", alpha=0.8)
                continue
            M29_eos_path = f"./{N}atoms/EoS_plots/eos_liquid_{N}atoms_{temp}K.txt"
            if os.path.exists(M29_eos_path): 
                M29_eos_data = np.loadtxt(M29_eos_path)
                # Assuming columns: rs, pressure, pressure_err
                axes[0].errorbar(M29_eos_data[:,0], M29_eos_data[:,1],xerr=M29_eos_data[:,2],yerr=M29_eos_data[:,3], fmt='--',label=f"M29 {N} atoms",alpha=0.8)
            DPA3_eos_path = f"../DPA3_dynamic/postprocessing/EOS/EOS.{temp}.{N}.dat"
            if os.path.exists(DPA3_eos_path):
                DPA3_eos_data = np.loadtxt(DPA3_eos_path)
                # Assuming columns: pressure, rs, rs_err
                #axes[0].errorbar(DPA3_eos_data[:,1], DPA3_eos_data[:,0], xerr=DPA3_eos_data[:,2], fmt='-', label=f"DPA3 {N} atoms",alpha=0.6)
            M29_cv_path = f"./{N}atoms/Cv_plots/Cv_liquid_{N}atoms_{temp}K_data.txt"
            if os.path.exists(M29_cv_path):
                M29_cv_data = np.loadtxt(M29_cv_path)
                # Assuming columns: Pressure, Cv, Cv_error
                axes[1].errorbar(M29_cv_data[:,0], M29_cv_data[:,1], yerr=M29_cv_data[:,2],fmt='-', label=f"M29 {N} atoms",alpha=1-i/10)
            DPA3_cv_path = f"../DPA3_dynamic/postprocessing/c/c.{temp}.{N}.dat"
            if os.path.exists(DPA3_cv_path):
                DPA3_cv_data = np.loadtxt(DPA3_cv_path)
                # Assuming columns: Pressure, Cv, Cv_error
                #axes[1].errorbar(DPA3_cv_data[:,0], DPA3_cv_data[:,1], yerr=DPA3_cv_data[:,2], fmt='s-', label=f"DPA3 {N} atoms")

        axes[0].set_xlabel("r$_s$ (bohr)")
        axes[0].set_ylabel("Pressure (GPa)")
        axes[0].set_title(f"EoS at {temp}K")
        axes[0].legend()
        axes[1].set_xlabel("Pressure (GPa)")
        axes[1].set_ylabel("Cv")
        #axes[1].set_xlim(140,180)
        #axes[1].set_ylim(bottom=0,top=0.6)
        axes[1].set_title(f"Cv at {temp}K")
        axes[1].legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f"./combined_plots/Cv_EoS_compare_{temp}K.png", dpi=300)
        plt.close()

def Cv_plot(atom_sizes,temps,case="NPT_mliap"):
       for temp in temps:
        # EoS plot
        for i,N in enumerate(atom_sizes):
            if N>1200:
                M29_eos_path = f"./Ilnur_data/eos_liquid_{N}atoms_{temp}K.txt"
                if os.path.exists(M29_eos_path):
                    EOS_data = np.loadtxt(M29_eos_path)
                    plt.errorbar(EOS_data[:,4],EOS_data[:,6],xerr=EOS_data[:,5],yerr=EOS_data[:,7],fmt='s',label=f"M29 {N} atoms",alpha=0.6)
                continue
            if temp==2700:
                continue
            if "ASE" in case:
                M29_cv_path = f"./{N}atoms/Cv_plots/ASE/Cv_liquid_{N}atoms_{temp}K_data.txt"
            else:
                M29_cv_path = f"./{N}atoms/Cv_plots/Cv_liquid_{N}atoms_{temp}K_data.txt"
            if temp==3000:
                M29_cv_path = f"./{N}atoms/Cv_plots/Cv_liquid_{N}atoms_{temp}K_data.txt"
            if os.path.exists(M29_cv_path):
                M29_cv_data = np.loadtxt(M29_cv_path)
                # Assuming columns: Pressure, Cv, Cv_error
                plt.errorbar(M29_cv_data[:,0], M29_cv_data[:,1], yerr=M29_cv_data[:,2],fmt='s', label=f"M29 {N} atoms",alpha=1-i/10)
            # DPA3_cv_path = f"../DPA3_dynamic/postprocessing/c/c.{temp}.{N}.dat"
            # if os.path.exists(DPA3_cv_path):
            #     DPA3_cv_data = np.loadtxt(DPA3_cv_path)
            #     # Assuming columns: Pressure, Cv, Cv_error
            #     #axes[1].errorbar(DPA3_cv_data[:,0], DPA3_cv_data[:,1], yerr=DPA3_cv_data[:,2], fmt='s-', label=f"DPA3 {N} atoms")

        plt.xlabel("Pressure (GPa)")
        plt.ylabel("Cv")
        #axes[1].set_xlim(140,180)
        #axes[1].set_ylim(bottom=0,top=0.6)
        plt.title(f"Cv at {temp}K")
        plt.legend(loc='upper left')
        plt.tight_layout()
        if "ASE" in case:
            plt.savefig(f"./combined_plots/ASE/Cv_ontop_{temp}K.png", dpi=300)
        else:
            plt.savefig(f"./combined_plots/Cv_ontop_{temp}K.png", dpi=300)
        plt.close()

def PV_eos_plot_ASE(atom_sizes,temps):
    for temp in temps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # EoS plot
        for i,N in enumerate(atom_sizes):
            if N>1200:
                M29_eos_path = f"./Ilnur_data/eos_liquid_{N}atoms_{temp}K.txt"
                if os.path.exists(M29_eos_path):
                    EOS_data = np.loadtxt(M29_eos_path)
                    if N==2592:
                        axes[0].errorbar(EOS_data[:,0], EOS_data[:,4], xerr=EOS_data[:,1], yerr=EOS_data[:,5], fmt='--', label=f"M29 {N} atoms", alpha=0.8)
                        axes[1].errorbar(EOS_data[:,4],EOS_data[:,-2],xerr=EOS_data[:,5],yerr=EOS_data[:,-1],fmt='s',label=f"M29 {N} atoms",alpha=0.6)
                    else:
                        axes[0].errorbar(EOS_data[:,0], EOS_data[:,2], xerr=EOS_data[:,1], yerr=EOS_data[:,3], fmt='--', label=f"M29 {N} atoms", alpha=0.8)
                continue
            M29_eos_path = f"./{N}atoms/EoS_plots/ASE/eos_liquid_{N}atoms_{temp}K.txt"
            if os.path.exists(M29_eos_path): 
                M29_eos_data = np.loadtxt(M29_eos_path)
                # Assuming columns: rs, pressure, pressure_err
                axes[0].errorbar(M29_eos_data[:,0], M29_eos_data[:,1],xerr=M29_eos_data[:,2],yerr=M29_eos_data[:,3], fmt='--',label=f"M29 {N} atoms",alpha=0.8)
            DPA3_eos_path = f"../DPA3_dynamic/postprocessing/EOS/EOS.{temp}.{N}.dat"
            if os.path.exists(DPA3_eos_path):
                DPA3_eos_data = np.loadtxt(DPA3_eos_path)
                # Assuming columns: pressure, rs, rs_err
                #axes[0].errorbar(DPA3_eos_data[:,1], DPA3_eos_data[:,0], xerr=DPA3_eos_data[:,2], fmt='-', label=f"DPA3 {N} atoms",alpha=0.6)
            M29_cv_path = f"./{N}atoms/Cv_plots/ASE/Cv_liquid_{N}atoms_{temp}K_data.txt"
            if os.path.exists(M29_cv_path):
                M29_cv_data = np.loadtxt(M29_cv_path)
                # Assuming columns: Pressure, Cv, Cv_error
                axes[1].errorbar(M29_cv_data[:,0], M29_cv_data[:,1], yerr=M29_cv_data[:,2],fmt='s', label=f"M29 {N} atoms",alpha=1-i/10)
            DPA3_cv_path = f"../DPA3_dynamic/postprocessing/c/c.{temp}.{N}.dat"
            if os.path.exists(DPA3_cv_path):
                DPA3_cv_data = np.loadtxt(DPA3_cv_path)
                # Assuming columns: Pressure, Cv, Cv_error
                #axes[1].errorbar(DPA3_cv_data[:,0], DPA3_cv_data[:,1], yerr=DPA3_cv_data[:,2], fmt='s-', label=f"DPA3 {N} atoms")

        axes[0].set_xlabel("r$_s$ (bohr)")
        axes[0].set_ylabel("Pressure (GPa)")
        axes[0].set_title(f"EoS at {temp}K")
        axes[0].legend()
        axes[1].set_xlabel("Pressure (GPa)")
        axes[1].set_ylabel("Cv")
        #axes[1].set_xlim(140,180)
        #axes[1].set_ylim(bottom=0,top=0.6)
        axes[1].set_title(f"Cv at {temp}K")
        axes[1].legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f"./combined_plots/ASE/Cv_EoS_compare_{temp}K.png", dpi=300)
        plt.close()

def main():
    atom_sizes = [360,576,1200,2592]  # Add more sizes as needed
    #temps = range(1700, 2501, 100)  # 1700K to 2400K, step 100K
    #Cv_plot(atom_sizes, temps, case="NPT_ASE")
    temps = [1700,1800,2000,2100,2300,2500,2700,3000]
    #PV_eos_plot_ASE(atom_sizes, temps)
    #PV_eos_plot(atom_sizes, temps)
    Cv_plot(atom_sizes, temps, case="NPT_mliap")
    #Cv_plot(atom_sizes, temps, case="NPT_ASE")
    
if __name__ == "__main__":
    main()