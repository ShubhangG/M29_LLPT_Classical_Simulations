import numpy as np
from ase import io
from natsort import natsorted
import sys
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen")
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack")
import os
import matplotlib.pyplot as plt
from qharv_db.grsk import calc_sofk_npt
from qharv_db import grsk
from qharv.inspect import axes_pos
import pickle as pk
import argparse

def get_gofr_norm(axes, bin_edges, n1, n2=None):
    ndim, ndim = axes.shape
    # calculate volume of bins
    vnorm = np.diff(2*(ndim-1)/ndim*np.pi*bin_edges**ndim)
    # calculate density normalization
    if n2 is None:
        npair = n1*(n1-1)/2
    else:
        npair = n1*n2
    rho = npair/abs(np.linalg.det(axes))
    # assemble the norm vector
    gr_norm = 1./(rho*vnorm)
    return gr_norm

def ase_gofr(atoms, bin_edges):
    """Calculate the real-space pair correlation function g(r) among
    all pairs of atom types. Histogram distances along the radial
    direction, i.e. spherically averaged.

    Args:
    atoms (ase.Atoms): atoms
    bin_edges (np.array): histogram bin edges
    gr_norm (np.array): normalization of each bin
    Return:
    dict: gr1_map, one g(r) for each pair of atom types.
    Example:
    >>> axes = np.eye(3)
    >>> pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> atoms = Atoms('H2', cell=axes, positions=pos, pbc=1)
    >>> bin_edges = get_bin_edges(axes)
    >>> gr_norm = get_gofr_norm(axes, bin_edges, len(pos))
    >>> gr1_map = ase_gofr(atoms, bin_edges, gr_norm)
    >>> gr1 = gr1_map[(0, 0)]
    >>> r = 0.5*(bin_edges[1:]+bin_edges[:-1])
    >>> plt.plot(r, gr1)
    >>> plt.show()
    """
    from ase.geometry import get_distances
    ias = np.unique(atoms.get_atomic_numbers())
    gr1_map = {}  # snapshot g(r) between all pairs of particle types
    for i in range(len(ias)):
        for j in range(i, len(ias)):
            ia = ias[i]
            ja = ias[j]
            # select positions
            idx1 = [atom.index for atom in atoms if atom.number == ia]
            idx2 = [atom.index for atom in atoms if atom.number == ja]
            ni = len(idx1)
            nj = len(idx2)
            # calculate distances
            drij, rij = get_distances(
            atoms[idx1].get_positions(),
            p2=atoms[idx2].get_positions(),
            cell=atoms.get_cell(),
            pbc=1
            )
            gr_norm = get_gofr_norm(np.array(atoms.get_cell()),bin_edges,ni)
            # extract unique distances
            offset = 0
            if ia == ja:
                offset = 1
                idx = np.triu_indices_from(rij, offset)
                dists = rij[idx]
            # histogram
            hist, be = np.histogram(dists, bin_edges)
            gr1 = hist*gr_norm
            gr1_map[(ia, ja)] = gr1
    return gr1_map

def gen_data_path(P,T,N,phase,case):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{P}/{phase}/{case}/{T}/analysis/"

def gen_path2(press,temp,N,phase,case):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/{N}atoms/p{press}/{phase}/{case}/{temp}/"


def what_time(file,file_start):
    time = file.removeprefix(file_start).removesuffix(".atom")
    if time == 'init':
        time = -1
    else:
        time = int(time)
    return(time)


def get_pos_and_axes(path_to_dump_file,format='lammps-dump-text'):
    posl= []
    axesl=[]
    # Read atoms
    traj = io.read(path_to_dump_file,format=format,index=':')
    # Get atoms coordinates
    for i, atoms in enumerate(traj):
        # axes
        axes  = atoms.get_cell()
        axesl.append(axes)
        # positions
        pos   = atoms.get_positions()
        posl.append(pos)
    return(axesl,posl)


def plot_sofk_for_all_traj(pressure,temperature,N,phase,case):
    path = gen_data_path(pressure,temperature,N,phase,case)
    if "ASE" in case:
        axesl,posl = get_pos_and_axes(path+f'P{pressure}T{temperature}.merged.traj',format="traj")
        atoms = io.read(path+f'P{pressure}T{temperature}.merged.traj')
    else:  
        axesl,posl = get_pos_and_axes(path+'dump.merged.atom')
        atoms = io.read(path+'dump.merged.atom',format='lammps-dump-text')
    axesl = axesl[-100:]
    posl = posl[-100:]
    N_protons = atoms.get_global_number_of_atoms()
    L_box= max(np.diag(atoms.get_cell()))
    kf=2.0
    # Choose bin 
    bin_edges = np.arange(1e-3,5*kf,2*np.pi/L_box) # Can modify this!
    nbins = len(bin_edges)
    # Get sofk
    uskm, uske = calc_sofk_npt(bin_edges,axesl,posl,nbins)
    delta= np.diff(bin_edges)
    plotting_bins = bin_edges[:len(uskm)]+delta/2
    sofk_save = np.c_[plotting_bins, uskm, uske]
    np.savetxt(path+f"P{pressure}T{temperature}{phase}sofk.txt",sofk_save)
    plt.errorbar(plotting_bins,uskm,yerr=uske,xerr=None)
    plt.title("N={}, L_box={}, {}K,{}GPa".\
        format(N_protons,round(L_box,3),temperature,pressure))
    plt.savefig(path+"N{}_P{}_T{}_L{}_Kf{}_sk.png"\
        .format(N_protons,pressure,temperature,round(L_box,2),kf))
    plt.show()
    print("N{}_P{}_T{}_L{}_Kf{}_sk.png"\
        .format(N_protons,pressure,temperature,round(L_box,3),kf))
    plt.close()

def plot_sofk_for_each_batch_run(pressure,temperature,N,phase,case):
    path = gen_path2(pressure,temperature,N,phase,case)
    all_data = natsorted(os.listdir(path))
    filtering_substring = [".atom"]
    simout_filelist = list(filter(lambda x: any(substring in x for substring in filtering_substring), all_data))
    if any('dump.init.atom' in x for x in simout_filelist):
        simout_filelist.remove("dump.init.atom")
        simout_filelist.insert(0,"dump.init.atom")
    for file_name in simout_filelist:
        axesl,posl = get_pos_and_axes(path+file_name)
        axesl = axesl[-100:]
        posl = posl[-100:]
        atoms = io.read(path+file_name)
        N_protons = atoms.get_global_number_of_atoms()
        L_box= max(np.diag(atoms.get_cell()))
        kf=2.0
        # Choose bin 
        bin_edges = np.arange(1e-3,5*kf,2*np.pi/L_box) # Can modify this!
        nbins = len(bin_edges)
    
        # Get sofk
        uskm, uske = calc_sofk_npt(bin_edges,axesl,posl,nbins)
        delta= np.diff(bin_edges)
        plotting_bins = bin_edges[:len(uskm)]+delta/2
        sofk_save = np.c_[plotting_bins, uskm, uske]
        np.savetxt(path+f"P{pressure}T{temperature}{phase}sofk_{file_name[:-5]}.txt",sofk_save)
    
        plt.errorbar(plotting_bins,uskm,yerr=uske,xerr=None)
        plt.title(f"N={N_protons}, {temperature}K,{pressure}GPa, {file_name}")
        plt.savefig(path+f"N{N_protons}_P{pressure}_T{temperature}_sk"+file_name[:-5]+".png")
        print(file_name+" sofk finished")
        plt.close()


def plot_sofk_for_each_batch_run_ASE(pressure,temperature,N,phase,case):
    path = gen_path2(pressure,temperature,N,phase,case)
    all_data = natsorted(os.listdir(path))
    filtering_substring = [".traj"]
    simout_filelist = list(filter(lambda x: any(substring in x for substring in filtering_substring), all_data))
    for file_name in simout_filelist:
        axesl,posl = get_pos_and_axes(path+file_name,format="traj")
        axesl = axesl[-100:]
        posl = posl[-100:]
        atoms = io.read(path+file_name)
        N_protons = atoms.get_global_number_of_atoms()
        L_box= max(np.diag(atoms.get_cell()))
        kf=2.0
        # Choose bin 
        bin_edges = np.arange(1e-3,5*kf,2*np.pi/L_box) # Can modify this!
        nbins = len(bin_edges)
    
        # Get sofk
        uskm, uske = calc_sofk_npt(bin_edges,axesl,posl,nbins)
        delta= np.diff(bin_edges)
        plotting_bins = bin_edges[:len(uskm)]+delta/2
        sofk_save = np.c_[plotting_bins, uskm, uske]
        np.savetxt(path+f"P{pressure}T{temperature}{phase}sofk_{file_name[:-5]}.txt",sofk_save)
        plt.errorbar(plotting_bins,uskm,yerr=uske,xerr=None)
        plt.title(f"N={N_protons}, {temperature}K,{pressure}GPa, {file_name}")
        plt.savefig(path+f"N{N_protons}_P{pressure}_T{temperature}_sk"+file_name[:-5]+".png")
        print(file_name+" sofk finished")
        plt.close()

def plot_sofk_one_traj(atoms,pressure,temperature,phase,case):
    atoms = [atoms,atoms.copy()]
    N_protons = atoms[0].get_global_number_of_atoms()
    L_box= max(np.diag(atoms[0].get_cell()))
    kf=2.0
    # Choose bin 
    bin_edges = np.arange(1e-3,5*kf,2*np.pi/L_box) # Can modify this!
    nbins = len(bin_edges)
    axesl = [atoms[0].get_cell(),atoms[1].get_cell()]
    posl=[atoms[0].get_positions(),atoms[1].get_positions()]
    uskm,uske = calc_sofk_npt(bin_edges,axesl,posl,nbins)
    plt.plot(bin_edges[:len(uskm)],uskm)
    plt.title("N={}, pure hcp {}K,{}GPa".\
        format(N_protons,temperature,pressure))
    plt.savefig(gen_data_path(pressure,temperature,N_protons,phase,case)+f"N{N_protons}_P{pressure}_T{temperature}_Kf{kf}_{phase}_sk_1frame.png")
    plt.show()
    print(f"N{N_protons}_P{pressure}_T{temperature}_Kf{kf}_{phase}_sk")
    plt.close()

def get_gofr(kdat,bin_partitions):
    num_trajs,num_atoms = np.shape(kdat)
    g_r = np.zeros((num_trajs,len(bin_partitions)-1))
    for idx,traj in enumerate(kdat):
        g_map = ase_gofr(traj,bin_partitions)
        g_r[idx,:] = g_map[1,1]
    mean_g_r = np.mean(g_r,axis=0)
    std_gr = np.std(g_r,axis=0)
    return g_r,mean_g_r,std_gr

def calculate_gr(P,T,N,phase,case):
    path = gen_data_path(P,T,N,phase,case)
    if "ASE" in case:
        atoms = io.read(path+f"P{P}T{T}.merged.traj",index='-100:')
    else:
        atoms = io.read(path+"dump.merged.atom",format='lammps-dump-text',index='-100:')
    N_protons = atoms[0].get_global_number_of_atoms()
    axesl=[]
    posl=[]
    for traj in atoms:
        axes = traj.get_cell()
        pos = traj.get_positions()
        axesl.append(axes)
        posl.append(pos)
    rmax = axes_pos.rwsc(axesl[-1])
    bin_edges = grsk.gofr_bin_edges(0, rmax, 501)
    grm, gre = grsk.calc_gofr(bin_edges, axesl, posl)    
    r = 0.5*(bin_edges[:-1]+bin_edges[1:])
    gofr_tot = np.c_[r, grm, gre]
    if "ASE" in case:
        np.savetxt(path+f"P{P}T{T}gofr_info_ASE_{phase}.npy",gofr_tot)
    else:    
        np.savetxt(path+f"P{P}T{T}gofr_info_LAMMPS_{phase}.npy",gofr_tot)
    Ang2bohr = 1.88973
    vol_list = [aa.get_volume() for aa in atoms]
    rs = ((np.mean(vol_list)/N_protons)/(4/3*np.pi))**(1/3)
    rs = rs*Ang2bohr
    plt.errorbar(r,grm,yerr=gre)
    plt.title(f"P={P} GPa T={T} K phase={phase}, rs={round(rs,2)} bohr")
    plt.xlabel("r")
    plt.ylabel(r"g(r)")
    plt.savefig(path+f"P{P}T{T}gofr.png")
    plt.close()

def plot_gofr_for_each_batchrun(pressure,temperature,N,phase,case):
    path = gen_path2(pressure,temperature,N,phase,case)
    all_data = natsorted(os.listdir(path))    
    filtering_substring = [".atom"]
    simout_filelist = list(filter(lambda x: any(substring in x for substring in filtering_substring), all_data))
    if any('dump.init.atom' in x for x in simout_filelist):
        simout_filelist.remove("dump.init.atom")
        simout_filelist.insert(0,"dump.init.atom")
    for file_name in simout_filelist:
        atoms = io.read(path+file_name,format='lammps-dump-text',index='-100:')
        N_protons = atoms[0].get_global_number_of_atoms()
        axesl=[]
        posl=[]
        for traj in atoms:
            axes = traj.get_cell()
            pos = traj.get_positions()
            axesl.append(axes)
            posl.append(pos)
        rmax = axes_pos.rwsc(axesl[-1])
        bin_edges = grsk.gofr_bin_edges(0, rmax, 51)
        grm, gre = grsk.calc_gofr(bin_edges, axesl, posl)    
        r = 0.5*(bin_edges[:-1]+bin_edges[1:])
        gofr_tot = np.c_[r, grm, gre]
        Ang2bohr = 1.88973
        vol_list = [aa.get_volume() for aa in atoms]
        rs = ((np.mean(vol_list)/N_protons)/(4/3*np.pi))**(1/3)
        rs = rs*Ang2bohr
        np.savetxt(path+f"P{pressure}T{temperature}{phase}gofr_{file_name[:-5]}.txt",gofr_tot)
        plt.errorbar(r,grm,yerr=gre)
        plt.title(f"P={pressure} GPa T={temperature} K phase={phase} rs={round(rs,2)} bohr")
        plt.xlabel("r")
        plt.ylabel(r"g(r)")
        plt.savefig(path+f"P{pressure}T{temperature}gofr_{file_name[:-5]}.png")
        plt.close()


def plot_gofr_for_each_batchrun_ASE(pressure,temperature,N,phase,case):
    path = gen_path2(pressure,temperature,N,phase,case)
    all_data = natsorted(os.listdir(path))    
    filtering_substring = [".traj"]
    simout_filelist = list(filter(lambda x: any(substring in x for substring in filtering_substring), all_data))
    for file_name in simout_filelist:
        atoms = io.read(path+file_name,index='-100:')
        N_protons = atoms[0].get_global_number_of_atoms()
        axesl=[]
        posl=[]
        for traj in atoms:
            axes = traj.get_cell()
            pos = traj.get_positions()
            axesl.append(axes)
            posl.append(pos)
        rmax = axes_pos.rwsc(axesl[-1])
        bin_edges = grsk.gofr_bin_edges(0, rmax, 51)
        grm, gre = grsk.calc_gofr(bin_edges, axesl, posl)    
        r = 0.5*(bin_edges[:-1]+bin_edges[1:])
        gofr_tot = np.c_[r, grm, gre]
        Ang2bohr = 1.88973
        vol_list = [aa.get_volume() for aa in atoms]
        rs = ((np.mean(vol_list)/N_protons)/(4/3*np.pi))**(1/3)
        rs = rs*Ang2bohr
        np.savetxt(path+f"P{pressure}T{temperature}{phase}gofr_{file_name[:-5]}.txt",gofr_tot)
        plt.errorbar(r,grm,yerr=gre)
        plt.title(f"P={pressure} GPa T={temperature} K phase={phase} rs={round(rs,2)} bohr")
        plt.xlabel("r")
        plt.ylabel(r"g(r)")
        plt.savefig(path+f"P{pressure}T{temperature}gofr_{file_name[:-5]}.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='input P-T to create a two-phase starting configuration from prev results')
    parser.add_argument('-p','--press',dest='pressure',type=int,help='pressure',required=True)
    parser.add_argument('-t','--temp',dest='temp',type=int,help='temperature',required=True)
    parser.add_argument('-f','--phase',dest='phase',type=str,help='solid or liquid phase',required=True)
    parser.add_argument('-c','--case',dest='case',type=str,help="Is it NVT or NPT simulation",default="NPT")
    parser.add_argument('-n','--natom',dest='natom',type=int,help="Number of atoms",required=True)
    parser.add_argument('-o','--option',dest='option',type=str,help='sofk or gofr',required=True)
    parser.add_argument('-a','--all',dest='all',action='store_true',help="Choose to run your sofk or gofr on all dump files")
    args= parser.parse_args()
    press = args.pressure
    temp = args.temp
    phase = args.phase
    option = args.option
    case = args.case
    all = args.all
    N_atoms = args.natom

    if option=="gofr":
        if all:
            if case=="NPT_ASE":
                plot_gofr_for_each_batchrun_ASE(press,temp,N_atoms,phase,case)
            else:
                plot_gofr_for_each_batchrun(press,temp,N_atoms,phase,case)
        else:
            calculate_gr(press,temp,N_atoms,phase,case)
    elif option=="sofk":
        if all:
            if case=="NPT_ASE":
                plot_sofk_for_each_batch_run_ASE(press,temp,N_atoms,phase,case)
            else:
                plot_sofk_for_each_batch_run(press,temp,N_atoms,phase,case)
        else:
            plot_sofk_for_all_traj(press,temp,N_atoms,phase,case)


if __name__ == '__main__':
    main()
